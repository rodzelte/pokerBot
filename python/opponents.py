"""
opponents.py — Opponent pool and reward shaping for competitive poker training.

Provides:
  - SelfPlayAgent: frozen copy of the main agent for self-play
  - OpponentPool: manages a roster of past checkpoints + random agent
  - RewardShaper: transforms raw payoffs with poker-aware bonuses/penalties
  - CurriculumScheduler: gradually transitions from easy to hard opponents

The key insight: training against RandomAgent teaches you to beat fish,
but training against yourself (and past versions of yourself) teaches you
to play balanced, exploitable strategies that work against thinking opponents.
"""
from __future__ import annotations

import copy
import os
import glob
import logging
import random as pyrandom
from dataclasses import dataclass, field

import numpy as np
import torch
from rlcard.agents import DQNAgent, RandomAgent

log = logging.getLogger("opponents")


# ─── Self-Play Agent ────────────────────────────────────────────────────────

class SelfPlayAgent:
    """
    A frozen snapshot of a DQN/NFSP agent that acts as an opponent.
    Uses eval_step (no exploration) so it plays its best learned strategy.
    """

    def __init__(self, agent, name: str = "self"):
        self.use_raw = False
        self.name = name
        # Deep-copy the Q-network weights only (not replay buffer — saves memory)
        self._q_net_state = copy.deepcopy(agent.q_estimator.qnet.state_dict())
        self._num_actions = agent.q_estimator.num_actions
        self._state_shape = agent.q_estimator.state_shape
        self._mlp_layers = [256, 256, 128]
        self._device = agent.q_estimator.device
        # Build a lightweight eval-only agent
        self._agent = DQNAgent(
            num_actions=self._num_actions,
            state_shape=self._state_shape,
            mlp_layers=self._mlp_layers,
            device=self._device,
        )
        self._agent.q_estimator.qnet.load_state_dict(self._q_net_state)
        self._agent.q_estimator.qnet.eval()

    def step(self, state):
        """Training step — but frozen agent doesn't learn, just acts."""
        action, _ = self._agent.eval_step(state)
        return action

    def eval_step(self, state):
        """Evaluation step — returns (action, probs)."""
        return self._agent.eval_step(state)


class NFSPSelfPlayAgent:
    """Frozen snapshot of an NFSP agent for self-play (uses average policy)."""

    def __init__(self, agent, name: str = "self-nfsp"):
        self.use_raw = False
        self.name = name
        self._policy_state = copy.deepcopy(agent.policy_network.state_dict())
        self._rl_state = copy.deepcopy(agent._rl_agent.q_estimator.qnet.state_dict())
        self._rl_target_state = copy.deepcopy(agent._rl_agent.target_estimator.qnet.state_dict())

        # Reconstruct a lightweight NFSP for eval
        from rlcard.agents import NFSPAgent
        self._agent = NFSPAgent(
            num_actions=agent._num_actions,
            state_shape=agent._state_shape,
            hidden_layers_sizes=[256, 256, 128],
            q_mlp_layers=[256, 256, 128],
            device=agent.device,
            evaluate_with="average_policy",
        )
        self._agent.policy_network.load_state_dict(self._policy_state)
        self._agent._rl_agent.q_estimator.qnet.load_state_dict(self._rl_state)
        self._agent._rl_agent.target_estimator.qnet.load_state_dict(self._rl_target_state)
        self._agent.policy_network.eval()

    def step(self, state):
        action, _ = self._agent.eval_step(state)
        return action

    def eval_step(self, state):
        return self._agent.eval_step(state)


# ─── Opponent Pool ───────────────────────────────────────────────────────────

@dataclass
class OpponentPool:
    """
    Maintains a pool of opponents of varying strength.
    
    The agent trains against a MIX of:
      - Current self (frozen periodically) → learns counter-strategies
      - Past checkpoints → doesn't forget how to beat older versions
      - Random agent → maintains ability to exploit weak players
    
    This diversity prevents overfitting to a single opponent style.
    """
    num_actions: int
    device: torch.device
    max_pool_size: int = 10
    _pool: list = field(default_factory=list)
    _names: list = field(default_factory=list)

    def __post_init__(self):
        # Always start with a random agent in the pool
        self._pool.append(RandomAgent(num_actions=self.num_actions))
        self._names.append("random")

    def add_snapshot(self, agent, name: str = "snapshot") -> None:
        """Add a frozen snapshot of the current agent to the pool."""
        try:
            # Detect if it's NFSP or DQN by checking for policy_network
            if hasattr(agent, "policy_network"):
                snapshot = NFSPSelfPlayAgent(agent, name=name)
            else:
                snapshot = SelfPlayAgent(agent, name=name)
            self._pool.append(snapshot)
            self._names.append(name)
            # Evict oldest non-random if over capacity
            if len(self._pool) > self.max_pool_size:
                self._pool.pop(1)  # index 0 is random, keep it
                self._names.pop(1)
            log.info(f"Opponent pool: added '{name}' (pool size: {len(self._pool)})")
        except Exception as e:
            log.warning(f"Failed to snapshot agent for pool: {e}")

    def add_from_checkpoint(self, checkpoint_path: str, state_shape, name: str = "ckpt") -> None:
        """Load a past checkpoint as a pool opponent."""
        if not os.path.exists(checkpoint_path):
            return
        try:
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            dummy = DQNAgent(
                num_actions=self.num_actions,
                state_shape=state_shape,
                mlp_layers=[256, 256, 128],
                device=self.device,
            )
            dummy.q_estimator.qnet.load_state_dict(ckpt["q_net"])
            snapshot = SelfPlayAgent(dummy, name=name)
            self._pool.append(snapshot)
            self._names.append(name)
            if len(self._pool) > self.max_pool_size:
                self._pool.pop(1)
                self._names.pop(1)
            log.info(f"Opponent pool: loaded '{name}' from {checkpoint_path}")
        except Exception as e:
            log.warning(f"Failed to load checkpoint opponent: {e}")

    def sample(self, weights: dict[str, float] | None = None) -> tuple:
        """
        Sample an opponent from the pool.
        
        weights: optional dict like {"random": 0.1, "self": 0.6, "ckpt": 0.3}
                 Keys are matched as prefixes against pool names.
        
        Returns (agent, name).
        """
        if len(self._pool) == 1:
            return self._pool[0], self._names[0]

        if weights:
            w = []
            for name in self._names:
                matched = False
                for prefix, weight in weights.items():
                    if name.startswith(prefix):
                        w.append(weight)
                        matched = True
                        break
                if not matched:
                    w.append(1.0)
            total = sum(w)
            w = [x / total for x in w]
            idx = pyrandom.choices(range(len(self._pool)), weights=w, k=1)[0]
        else:
            idx = pyrandom.randint(0, len(self._pool) - 1)

        return self._pool[idx], self._names[idx]

    @property
    def size(self) -> int:
        return len(self._pool)

    @property
    def names(self) -> list[str]:
        return list(self._names)


# ─── Stack Tracker ───────────────────────────────────────────────────────────

class StackTracker:
    """
    Virtual bankroll tracker across training episodes.

    Simulates a real poker session where you carry a stack across hands.
    When the stack drops too low, simulates a "rebuy" to initial stack.

    This teaches the agent:
    - Deep stack play: exploit positional advantage with bigger stacks
    - Short stack play: adjusted strategy when running low
    - Bankroll management: don't punt your stack on marginal spots
    - Equity awareness: the more BBs you have, the more flexibility you have
    """

    def __init__(self, initial_stack_bb: float = 100.0, rebuy_threshold_bb: float = 10.0,
                 rebuy_penalty: float = 0.15):
        self.initial_stack_bb = initial_stack_bb
        self.stack_bb = initial_stack_bb
        self.peak_bb = initial_stack_bb
        self.hands_played = 0
        self.win_streak = 0
        self.total_profit_bb = 0.0
        self.rebuy_threshold_bb = rebuy_threshold_bb
        self.rebuy_penalty = rebuy_penalty
        self.rebuys = 0

    def update(self, payoff: float) -> dict:
        """
        Update stack with hand result.

        Args:
            payoff: normalized payoff from env.run() in [-1, 1]

        Returns:
            dict with stack metrics for reward shaping
        """
        delta_bb = payoff * self.initial_stack_bb
        self.stack_bb += delta_bb
        self.total_profit_bb += delta_bb
        self.hands_played += 1

        if delta_bb > 0:
            self.win_streak += 1
        else:
            self.win_streak = 0

        if self.stack_bb > self.peak_bb:
            self.peak_bb = self.stack_bb

        # Auto-rebuy if busted — flag it so reward shaper can penalize
        rebuyed = False
        if self.stack_bb < self.rebuy_threshold_bb:
            self.stack_bb = self.initial_stack_bb
            self.rebuys += 1
            rebuyed = True

        return {
            "stack_bb": self.stack_bb,
            "delta_bb": delta_bb,
            "growth_ratio": self.stack_bb / self.initial_stack_bb,
            "from_peak": self.stack_bb / max(self.peak_bb, 0.01),
            "win_streak": self.win_streak,
            "total_profit_bb": self.total_profit_bb,
            "hands_played": self.hands_played,
            "rebuys": self.rebuys,
            "rebuyed": rebuyed,
        }

    def summary(self) -> str:
        bb_per_hand = self.total_profit_bb / max(self.hands_played, 1)
        return (
            f"Stack: {self.stack_bb:.1f}BB | "
            f"Profit: {self.total_profit_bb:+.1f}BB | "
            f"BB/hand: {bb_per_hand:+.3f} | "
            f"Peak: {self.peak_bb:.1f}BB | "
            f"Rebuys: {self.rebuys}"
        )


# ─── Curriculum Scheduler ────────────────────────────────────────────────────

class CurriculumScheduler:
    """
    Gradually shifts opponent mix from easy to hard:
    
    Phase 1 (warmup):  100% random  — learn basic hand values, don't fold everything
    Phase 2 (early):   70% random, 30% self-play  — start learning against yourself
    Phase 3 (mid):     20% random, 50% self-play, 30% pool  — diverse opponents
    Phase 4 (late):    10% random, 40% self-play, 50% pool  — mostly strong opponents
    """

    def __init__(self, total_episodes: int, warmup_pct: float = 0.05):
        self.total = total_episodes
        self.warmup_end = int(total_episodes * warmup_pct)
        self.early_end = int(total_episodes * 0.20)
        self.mid_end = int(total_episodes * 0.60)

    def get_phase(self, episode: int) -> str:
        if episode < self.warmup_end:
            return "warmup"
        elif episode < self.early_end:
            return "early"
        elif episode < self.mid_end:
            return "mid"
        else:
            return "late"

    def get_opponent_weights(self, episode: int) -> dict[str, float]:
        """Return sampling weights for the opponent pool."""
        phase = self.get_phase(episode)
        if phase == "warmup":
            return {"random": 1.0, "self": 0.0, "snap": 0.0, "ckpt": 0.0}
        elif phase == "early":
            return {"random": 0.7, "self": 0.3, "snap": 0.2, "ckpt": 0.1}
        elif phase == "mid":
            return {"random": 0.2, "self": 0.5, "snap": 0.3, "ckpt": 0.3}
        else:  # late
            return {"random": 0.1, "self": 0.4, "snap": 0.3, "ckpt": 0.5}

    def should_snapshot(self, episode: int, interval: int = 10000) -> bool:
        """Whether to take a snapshot of the agent for the opponent pool."""
        if episode < self.warmup_end:
            return False
        return episode % interval == 0


# ─── Reward Shaper ───────────────────────────────────────────────────────────

class RewardShaper:
    """
    Transform raw game payoffs with poker-aware shaping signals.

    The raw rlcard reward is simply +/- chips won. This enhances it with
    training signals that teach realistic poker fundamentals:

    Core signals:
    - Loss scaling: losses weighted heavier than wins (risk aversion)
    - Survival bonus: reward for seeing more streets (patience/info gathering)

    Stack-aware signals (when stack_metrics provided via StackTracker):
    - Big pot premium: winning large pots is extra valuable (pot building)
    - Stack growth bonus: growing above initial stack earns bonus
    - Deep stack bonus: maintaining 80+ BB earns ongoing reward
    - Short stack urgency: wins below 30 BB get extra reward (comeback)
    - Stack protection: extra loss penalty when stack is declining
    - Win streak bonus: consecutive wins earn momentum reward
    - Rebuy penalty: each rebuy applies escalating punishment (going broke is BAD)

    All bonuses are small relative to actual payoff to preserve the
    true objective (win chips / grow stack).
    """

    def __init__(
        self,
        loss_multiplier: float = 1.5,
        street_bonus: float = 0.02,
        survival_bonus: float = 0.01,
        big_pot_bonus: float = 0.15,
        stack_growth_bonus: float = 0.05,
        deep_stack_bonus: float = 0.02,
        short_stack_urgency: float = 0.03,
        stack_protection: float = 0.3,
        streak_bonus: float = 0.03,
        rebuy_penalty: float = 0.15,
        enabled: bool = True,
    ):
        self.loss_multiplier = loss_multiplier
        self.street_bonus = street_bonus
        self.survival_bonus = survival_bonus
        self.big_pot_bonus = big_pot_bonus
        self.stack_growth_bonus = stack_growth_bonus
        self.deep_stack_bonus = deep_stack_bonus
        self.short_stack_urgency = short_stack_urgency
        self.stack_protection = stack_protection
        self.streak_bonus = streak_bonus
        self.rebuy_penalty = rebuy_penalty
        self.enabled = enabled

    def shape(self, payoffs, trajectories, stack_metrics=None):
        """
        Apply reward shaping to raw payoffs.

        Args:
            payoffs: raw payoffs from env.run() [hero_payoff, villain_payoff]
            trajectories: raw trajectories from env.run()
            stack_metrics: optional dict from StackTracker.update() for
                           stack-aware BB-relative reward shaping

        Returns:
            shaped payoffs (same shape as input)
        """
        if not self.enabled:
            return list(payoffs)

        shaped = list(payoffs)

        for i in range(len(shaped)):
            raw = shaped[i]

            # Asymmetric loss scaling: penalize losses more heavily
            if raw < 0:
                shaped[i] = raw * self.loss_multiplier

            # Street survival bonus: more actions = more streets = small bonus
            if trajectories and i < len(trajectories):
                num_actions = len(trajectories[i])
                shaped[i] += min(num_actions * self.survival_bonus, 0.1)

            # ── Stack-aware shaping (hero only, i == 0) ──
            if stack_metrics and i == 0:
                # Big pot premium: winning large amounts is extra valuable
                # Teaches the agent to build pots when it has equity
                if raw > 0:
                    pot_premium = min(raw ** 2 * self.big_pot_bonus, 0.2)
                    shaped[i] += pot_premium

                # Stack growth bonus: reward for being above starting stack
                growth = stack_metrics.get("growth_ratio", 1.0)
                if growth > 1.0:
                    shaped[i] += self.stack_growth_bonus * min(growth - 1.0, 2.0)
                elif growth < 0.5 and raw < 0:
                    # Stack declining badly — amplify loss signal
                    shaped[i] *= 1.0 + self.stack_protection * (1.0 - growth)

                # Deep stack bonus: maintaining a healthy stack
                stack_bb = stack_metrics.get("stack_bb", 100.0)
                if stack_bb > 80:
                    shaped[i] += self.deep_stack_bonus
                elif stack_bb < 30 and raw > 0:
                    # Short stack win — extra urgency reward
                    shaped[i] += self.short_stack_urgency

                # Win streak momentum
                streak = stack_metrics.get("win_streak", 0)
                if streak >= 3:
                    shaped[i] += self.streak_bonus * min(streak, 10) / 10.0

                # Rebuy penalty: going broke is the worst outcome in poker.
                # Each rebuy applies escalating punishment — the more rebuys,
                # the harsher the penalty. Teaches the agent to protect its stack.
                if stack_metrics.get("rebuyed", False):
                    total_rebuys = stack_metrics.get("rebuys", 1)
                    # Escalating: 1st rebuy = -0.15, 2nd = -0.30, 3rd = -0.45 ...
                    shaped[i] -= self.rebuy_penalty * total_rebuys

        return shaped


# ─── Helper: snapshot interval based on curriculum ───────────────────────────

def get_snapshot_interval(total_episodes: int) -> int:
    """How often to snapshot the agent to the opponent pool."""
    if total_episodes <= 50000:
        return 5000
    elif total_episodes <= 200000:
        return 10000
    elif total_episodes <= 1000000:
        return 25000
    else:
        return 50000


# ─── Noisy Action Wrapper ────────────────────────────────────────────────────

class NoisyActionWrapper:
    """
    Action-biased exploration for realistic poker training.

    Standard epsilon-greedy treats all random actions equally, but in poker:
    - Random folds teach nothing (you forfeit the hand and information)
    - Random calls/raises teach about pot dynamics and opponent ranges

    This wrapper biases exploration toward aggressive/creative plays:
    1. Produces "human-like" variance (calling too wide, occasional bluffs)
    2. Teaches the agent about pot building and stack commitment
    3. Creates diverse training data instead of fold-heavy trajectories

    The noise decays over training so the agent converges toward optimal play.
    """

    def __init__(self, agent, noise_rate: float = 0.08,
                 aggression_bias: float = 0.6,
                 min_noise: float = 0.02,
                 total_episodes: int = 1):
        self.agent = agent
        self.noise_rate = noise_rate
        self.initial_noise = noise_rate
        self.min_noise = min_noise
        self.aggression_bias = aggression_bias
        self.total_episodes = max(total_episodes, 1)
        self.use_raw = getattr(agent, 'use_raw', False)

    def update_noise(self, episode: int) -> None:
        """Decay noise linearly over training."""
        progress = min(episode / self.total_episodes, 1.0)
        self.noise_rate = self.initial_noise - progress * (self.initial_noise - self.min_noise)

    def step(self, state):
        """Select action with poker-aware exploration noise."""
        if pyrandom.random() < self.noise_rate:
            legal_actions = list(state['legal_actions'].keys())
            if not legal_actions:
                return self.agent.step(state)

            if pyrandom.random() < self.aggression_bias:
                # Bias toward aggressive actions (higher index = bigger bet)
                mid = len(legal_actions) // 2
                aggressive = legal_actions[mid:] or legal_actions
                return pyrandom.choice(aggressive)
            else:
                # Occasionally passive (check/call)
                mid = max(len(legal_actions) // 2, 1)
                passive = legal_actions[:mid]
                return pyrandom.choice(passive)

        return self.agent.step(state)

    def eval_step(self, state):
        """Evaluation uses the agent's pure policy (no noise)."""
        return self.agent.eval_step(state)

    def feed(self, ts):
        """Delegate learning to the wrapped agent."""
        self.agent.feed(ts)


# ─── Stack-Aware Evaluation ──────────────────────────────────────────────────

def stack_tournament(env, num_games: int, initial_stack_bb: float = 100.0) -> dict:
    """
    Evaluate agent with a running stack across hands (simulates a real session).

    Unlike standard tournament() which just averages payoffs, this simulates
    a real poker session where chips carry over. This measures the agent's
    ability to grow and manage a stack — the ultimate poker skill.

    Returns dict with:
        profit_bb:      total profit/loss in big blinds
        avg_payoff:     average per-game payoff (same as tournament())
        final_stack_bb: final stack size in BB
        peak_bb:        highest stack reached
        hands_won:      number of profitable hands
        win_rate_pct:   percentage of hands won
    """
    stack_bb = initial_stack_bb
    peak_bb = initial_stack_bb
    total_payoff = 0.0
    hands_won = 0
    rebuys = 0

    for _ in range(num_games):
        _, payoffs = env.run(is_training=False)
        payoff = payoffs[0]
        total_payoff += payoff

        delta_bb = payoff * initial_stack_bb
        stack_bb += delta_bb
        if delta_bb > 0:
            hands_won += 1
        peak_bb = max(peak_bb, stack_bb)

        # Rebuy if busted
        if stack_bb < 5:
            stack_bb = initial_stack_bb
            rebuys += 1

    avg_payoff = total_payoff / max(num_games, 1)
    profit_bb = stack_bb - initial_stack_bb

    # Penalize eval score for rebuys: each rebuy costs 0.05 from avg_payoff
    # This makes an agent that wins a lot but goes broke often score lower
    # than one that wins steadily without ever rebuying.
    rebuy_penalty = rebuys * 0.05
    adjusted_payoff = avg_payoff - rebuy_penalty

    return {
        "profit_bb": profit_bb,
        "avg_payoff": adjusted_payoff,
        "raw_avg_payoff": avg_payoff,
        "final_stack_bb": stack_bb,
        "peak_bb": peak_bb,
        "hands_won": hands_won,
        "total_hands": num_games,
        "win_rate_pct": hands_won / max(num_games, 1) * 100,
        "rebuys": rebuys,
    }
