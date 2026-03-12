"""
deep_cfr_agent.py — Deep Counterfactual Regret Minimization for No-Limit Hold'em.

Deep CFR (Brown et al., 2019) replaces the tabular regret/strategy storage
of vanilla CFR with neural networks, allowing the algorithm to scale to
games with massive state spaces like NL Hold'em.

Architecture:
  - Advantage networks (one per player): approximate cumulative regret
  - Strategy network (shared): approximates the average strategy over all iterations
  - Reservoir buffers: store training data with weighted reservoir sampling

Algorithm:
  For T iterations:
    1. K tree traversals with external sampling (like MCCFR)
    2. During traversal, collect (state, advantages) for hero, (state, strategy) for opponent
    3. Train advantage networks on collected data
  After T iterations:
    Train strategy network on all collected strategy data → use for inference.

The strategy network IS the lookup table — it generalizes to unseen states,
making it viable for full NL Hold'em where tabular methods would need
billions of entries.
"""
from __future__ import annotations

import os
import logging
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from rlcard.utils import remove_illegal

log = logging.getLogger("deep_cfr")


# ─── Reservoir Buffer ────────────────────────────────────────────────────────

class ReservoirBuffer:
    """
    Fixed-capacity buffer with reservoir sampling.

    As training progresses, newer data is weighted higher (linear CFR weighting).
    Reservoir sampling ensures the buffer is representative of the full data
    distribution even when we've seen far more samples than the capacity.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: list[tuple[np.ndarray, np.ndarray, int]] = []
        self._count = 0

    def add(self, obs: np.ndarray, target: np.ndarray, iteration: int):
        """Add a (state, target, iteration_weight) sample."""
        entry = (obs.copy(), target.copy(), iteration)
        self._count += 1
        if len(self.buffer) < self.capacity:
            self.buffer.append(entry)
        else:
            # Reservoir sampling: replace random element
            j = np.random.randint(0, self._count)
            if j < self.capacity:
                self.buffer[j] = entry

    def sample_batch(self, batch_size: int):
        """Sample a random batch. Returns (obs, targets, weights) arrays."""
        n = min(batch_size, len(self.buffer))
        indices = np.random.choice(len(self.buffer), size=n, replace=False)
        obs = np.array([self.buffer[i][0] for i in indices], dtype=np.float32)
        targets = np.array([self.buffer[i][1] for i in indices], dtype=np.float32)
        weights = np.array([self.buffer[i][2] for i in indices], dtype=np.float32)
        return obs, targets, weights

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()
        self._count = 0


# ─── Neural Networks ─────────────────────────────────────────────────────────

class AdvantageNetwork(nn.Module):
    """Approximates cumulative advantage (regret) for a single player."""

    def __init__(self, state_dim: int, num_actions: int,
                 hidden_layers: list[int] | None = None):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [256, 256, 128]

        layers = []
        in_dim = state_dim
        for h in hidden_layers:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        layers.append(nn.Linear(in_dim, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class StrategyNetwork(nn.Module):
    """Approximates the average strategy (action probabilities)."""

    def __init__(self, state_dim: int, num_actions: int,
                 hidden_layers: list[int] | None = None):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [256, 256, 128]

        layers = []
        in_dim = state_dim
        for h in hidden_layers:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        layers.append(nn.Linear(in_dim, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # raw logits; softmax applied externally


# ─── Deep CFR Agent ──────────────────────────────────────────────────────────

class DeepCFRAgent:
    """
    Deep CFR: neural-network-based CFR for large games.

    Uses external sampling (like MCCFR) for tree traversal, but stores
    (state, advantage) data in reservoir buffers and trains neural nets
    to approximate the regret function. This lets CFR work on games where
    the number of information sets is too large for a lookup table.

    After training, the strategy network provides a complete poker strategy
    that generalizes to unseen game states.
    """

    def __init__(
        self,
        env,
        model_path: str = "./models/deep_cfr",
        device: str = "cpu",
        hidden_layers: list[int] | None = None,
        advantage_buffer_size: int = 1_000_000,
        strategy_buffer_size: int = 1_000_000,
        batch_size: int = 2048,
        train_steps: int = 4000,
        lr: float = 0.001,
    ):
        self.use_raw = False
        self.env = env
        self.model_path = model_path
        self.device = torch.device(device)
        self.num_actions = env.num_actions
        self.num_players = env.num_players
        self.batch_size = batch_size
        self.train_steps = train_steps
        self.lr = lr
        self.iteration = 0

        # Resolve state dimension
        ss = env.state_shape[0]
        self.state_dim = ss[0] if isinstance(ss, (list, tuple)) else ss

        if hidden_layers is None:
            hidden_layers = [256, 256, 128]

        # One advantage network per player
        self.advantage_nets = [
            AdvantageNetwork(self.state_dim, self.num_actions, hidden_layers).to(self.device)
            for _ in range(self.num_players)
        ]

        # Single strategy network (trained at the end)
        self.strategy_net = StrategyNetwork(
            self.state_dim, self.num_actions, hidden_layers
        ).to(self.device)

        # Reservoir buffers
        self.advantage_buffers = [
            ReservoirBuffer(advantage_buffer_size) for _ in range(self.num_players)
        ]
        self.strategy_buffer = ReservoirBuffer(strategy_buffer_size)

    def train(self, num_traversals: int = 1):
        """
        One Deep CFR outer iteration:
        1. K traversals collecting advantage + strategy data
        2. Train advantage networks
        """
        self.iteration += 1

        for _ in range(num_traversals):
            for player_id in range(self.num_players):
                self.env.reset()
                self._traverse(player_id)

        # Train advantage networks on buffered data
        losses = []
        for pid in range(self.num_players):
            if len(self.advantage_buffers[pid]) >= self.batch_size:
                loss = self._train_network(
                    self.advantage_nets[pid], self.advantage_buffers[pid]
                )
                losses.append(loss)
        return np.mean(losses) if losses else 0.0

    def _traverse(self, player_id):
        """External sampling traversal collecting training data."""
        if self.env.is_over():
            return np.array(self.env.get_payoffs(), dtype=np.float64)

        current_player = self.env.get_player_id()
        state = self.env.get_state(current_player)
        obs = state["obs"].flatten().astype(np.float32)
        legal_actions = list(state["legal_actions"].keys())

        if not legal_actions:
            return np.zeros(self.num_players)

        # Get advantages from the current player's advantage network
        strategy = self._get_strategy(obs, legal_actions, current_player)

        if current_player == player_id:
            # HERO: traverse all legal actions
            action_values = {}
            for action in legal_actions:
                self.env.step(action)
                action_values[action] = self._traverse(player_id)
                self.env.step_back()

            # Compute state value under current strategy
            state_value = np.zeros(self.num_players)
            for a in legal_actions:
                state_value += strategy[a] * action_values[a]

            # Build full advantage vector
            adv_vector = np.zeros(self.num_actions, dtype=np.float32)
            for a in legal_actions:
                adv_vector[a] = action_values[a][player_id] - state_value[player_id]

            # Store in advantage buffer
            self.advantage_buffers[player_id].add(obs, adv_vector, self.iteration)

            return state_value

        else:
            # OPPONENT: store strategy data + sample one action
            strat_vector = np.zeros(self.num_actions, dtype=np.float32)
            for a in legal_actions:
                strat_vector[a] = strategy[a]
            self.strategy_buffer.add(obs, strat_vector, self.iteration)

            # Sample action from strategy
            legal_probs = np.array([strategy[a] for a in legal_actions])
            total = legal_probs.sum()
            if total <= 0:
                legal_probs = np.ones(len(legal_actions)) / len(legal_actions)
            else:
                legal_probs /= total

            sampled = np.random.choice(legal_actions, p=legal_probs)
            self.env.step(sampled)
            result = self._traverse(player_id)
            self.env.step_back()
            return result

    def _get_strategy(self, obs: np.ndarray, legal_actions: list[int],
                      player_id: int) -> dict[int, float]:
        """Get strategy via regret matching on advantage network output."""
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            advantages = self.advantage_nets[player_id](obs_t).cpu().numpy().flatten()

        # Regret matching: clamp negatives, normalise over legal actions
        strategy = {}
        positive_sum = sum(max(0.0, advantages[a]) for a in legal_actions)

        if positive_sum > 0:
            for a in legal_actions:
                strategy[a] = max(0.0, advantages[a]) / positive_sum
        else:
            uniform = 1.0 / len(legal_actions)
            for a in legal_actions:
                strategy[a] = uniform

        return strategy

    def _train_network(self, network: nn.Module, buffer: ReservoirBuffer) -> float:
        """Train a network on reservoir buffer data. Returns final loss."""
        optimizer = optim.Adam(network.parameters(), lr=self.lr)
        network.train()
        final_loss = 0.0

        for _ in range(self.train_steps):
            obs_b, target_b, weight_b = buffer.sample_batch(self.batch_size)

            obs_t = torch.FloatTensor(obs_b).to(self.device)
            target_t = torch.FloatTensor(target_b).to(self.device)
            weight_t = torch.FloatTensor(weight_b).to(self.device)

            # Normalise weights so loss scale is independent of iteration
            weight_t = weight_t / (weight_t.sum() + 1e-8)

            predictions = network(obs_t)
            # Weighted MSE loss
            loss = (weight_t.unsqueeze(1) * (predictions - target_t) ** 2).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        network.eval()
        return final_loss

    def train_strategy_network(self):
        """
        Train the final strategy network on all collected strategy data.
        Call this ONCE after all outer iterations are done.
        """
        if len(self.strategy_buffer) < self.batch_size:
            log.warning(
                f"Only {len(self.strategy_buffer)} strategy samples collected "
                f"(need at least {self.batch_size}). Need more iterations."
            )
            # Train on whatever we have
            if len(self.strategy_buffer) == 0:
                return

        loss = self._train_network(self.strategy_net, self.strategy_buffer)
        log.info(
            f"Strategy network trained on {len(self.strategy_buffer)} samples, "
            f"final loss: {loss:.6f}"
        )

    def step(self, state):
        """Select action (for use as opponent in rlcard env)."""
        action, _ = self.eval_step(state)
        return action

    def eval_step(self, state):
        """
        Use strategy network for inference.
        Returns (action_id, probs_array).
        """
        obs = state["obs"].flatten().astype(np.float32)
        legal_actions = list(state["legal_actions"].keys())

        if not legal_actions:
            probs = np.zeros(self.num_actions)
            return 0, probs

        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            logits = self.strategy_net(obs_t).cpu().numpy().flatten()

        # Masked softmax over legal actions
        probs = np.zeros(self.num_actions, dtype=np.float64)
        legal_logits = np.array([logits[a] for a in legal_actions], dtype=np.float64)
        legal_logits -= legal_logits.max()  # numerical stability
        exp_logits = np.exp(legal_logits)
        legal_probs = exp_logits / (exp_logits.sum() + 1e-8)

        for i, a in enumerate(legal_actions):
            probs[a] = legal_probs[i]

        action = np.random.choice(self.num_actions, p=probs)
        return action, probs

    def save(self):
        """Save all networks and metadata."""
        os.makedirs(self.model_path, exist_ok=True)
        path = os.path.join(self.model_path, "deep_cfr_model.pt")
        torch.save(
            {
                "advantage_nets": [net.state_dict() for net in self.advantage_nets],
                "strategy_net": self.strategy_net.state_dict(),
                "iteration": self.iteration,
                "num_actions": self.num_actions,
                "state_dim": self.state_dim,
                "num_players": self.num_players,
                "agent_type": "deep_cfr",
                "advantage_buffer_sizes": [len(b) for b in self.advantage_buffers],
                "strategy_buffer_size": len(self.strategy_buffer),
            },
            path,
        )
        log.info(
            f"Deep CFR saved: {self.iteration} iterations, "
            f"strategy buffer: {len(self.strategy_buffer)} -> {path}"
        )

    def load(self):
        """Load networks from disk. Returns True if successful."""
        path = os.path.join(self.model_path, "deep_cfr_model.pt")
        if not os.path.exists(path):
            return False
        data = torch.load(path, map_location=self.device, weights_only=False)
        for pid, sd in enumerate(data["advantage_nets"]):
            self.advantage_nets[pid].load_state_dict(sd)
            self.advantage_nets[pid].eval()
        self.strategy_net.load_state_dict(data["strategy_net"])
        self.strategy_net.eval()
        self.iteration = data.get("iteration", 0)
        log.info(f"Deep CFR loaded: {self.iteration} iterations")
        return True
