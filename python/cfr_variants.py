"""
cfr_variants.py — CFR+ and Monte Carlo CFR (MCCFR) agents for rlcard.

CFR+ (Tammelin, 2014):
  Faster convergence than vanilla CFR by flooring negative cumulative regrets
  to zero after each update. This prevents the "regret drag" effect where
  early mistakes slow down convergence for millions of iterations.

External Sampling MCCFR (Lanctot et al., 2009):
  Instead of traversing ALL opponent actions at every decision point, samples
  one action from the current strategy. This dramatically reduces per-iteration
  cost, making CFR feasible for larger games. Each iteration is much cheaper
  but you need more of them — a great tradeoff for speed.

Both agents are compatible with rlcard environments and produce tabular
strategies that can be saved/loaded as pickle files.
"""
from __future__ import annotations

import collections
import os
import pickle
import logging

import numpy as np
from rlcard.utils import remove_illegal

log = logging.getLogger("cfr_variants")


# ─── CFR+ Agent ──────────────────────────────────────────────────────────────

class CFRPlusAgent:
    """
    CFR+ (Regret Matching+) — converges faster than vanilla CFR.

    Two key improvements over vanilla CFR:
    1. After each iteration, negative cumulative regrets are floored to 0.
       This means the agent "forgets" bad actions faster and commits to
       good ones sooner.
    2. Linear strategy averaging: iteration t's strategy counts t times
       more than iteration 1's. Recent iterations are better — weight them more.

    The combination typically achieves the same convergence in ~10x fewer
    iterations than vanilla CFR.
    """

    def __init__(self, env, model_path="./models/cfr_plus"):
        self.use_raw = False
        self.env = env
        self.model_path = model_path

        self.policy = collections.defaultdict(lambda: np.zeros(env.num_actions))
        self.average_policy = collections.defaultdict(lambda: np.zeros(env.num_actions))
        self.regrets = collections.defaultdict(lambda: np.zeros(env.num_actions))
        self.iteration = 0

    def train(self):
        """One full CFR+ iteration (traverse tree for each player)."""
        self.iteration += 1
        for player_id in range(self.env.num_players):
            self.env.reset()
            probs = np.ones(self.env.num_players)
            self._traverse(probs, player_id)
        self._update_policy()

    def _traverse(self, probs, player_id):
        """Traverse game tree, updating regrets with CFR+ floor."""
        if self.env.is_over():
            return self.env.get_payoffs()

        current_player = self.env.get_player_id()
        obs, legal_actions = self._get_state(current_player)
        action_probs = self._action_probs(obs, legal_actions, self.policy)

        action_utilities = {}
        state_utility = np.zeros(self.env.num_players)

        for action in legal_actions:
            action_prob = action_probs[action]
            new_probs = probs.copy()
            new_probs[current_player] *= action_prob

            self.env.step(action)
            utility = self._traverse(new_probs, player_id)
            self.env.step_back()

            state_utility += action_prob * utility
            action_utilities[action] = utility

        if current_player != player_id:
            return state_utility

        # Update regrets and average policy for traversing player
        player_prob = probs[current_player]
        counterfactual_prob = (np.prod(probs[:current_player]) *
                               np.prod(probs[current_player + 1:]))
        player_state_utility = state_utility[current_player]

        for action in legal_actions:
            regret = counterfactual_prob * (
                action_utilities[action][current_player] - player_state_utility
            )
            # CFR+ key: floor cumulative regret to 0
            self.regrets[obs][action] = max(0.0, self.regrets[obs][action] + regret)

            # Linear averaging: weight by iteration number
            action_prob = action_probs[action]
            self.average_policy[obs][action] += self.iteration * player_prob * action_prob

        return state_utility

    def _update_policy(self):
        """Update current policy from non-negative regrets."""
        for obs in self.regrets:
            self.policy[obs] = self._regret_matching_plus(obs)

    def _regret_matching_plus(self, obs):
        """Regret matching on non-negative regrets (always well-defined)."""
        regret = self.regrets[obs]
        positive_sum = np.sum(regret)

        action_probs = np.zeros(self.env.num_actions)
        if positive_sum > 0:
            action_probs = regret / positive_sum
        else:
            action_probs[:] = 1.0 / self.env.num_actions
        return action_probs

    def _action_probs(self, obs, legal_actions, policy):
        """Get action probabilities, masking illegal actions."""
        if obs not in policy:
            action_probs = np.ones(self.env.num_actions) / self.env.num_actions
            self.policy[obs] = action_probs.copy()
        else:
            action_probs = policy[obs]
        return remove_illegal(action_probs, legal_actions)

    def _get_state(self, player_id):
        """Get (obs_key, legal_actions) for a player."""
        state = self.env.get_state(player_id)
        return state["obs"].tobytes(), list(state["legal_actions"].keys())

    def step(self, state):
        """Select action (for use as opponent in rlcard env)."""
        action, _ = self.eval_step(state)
        return action

    def eval_step(self, state):
        """Select action from average policy. Returns (action, probs_array)."""
        obs = state["obs"].tobytes()
        legal_actions = list(state["legal_actions"].keys())
        probs = self._action_probs(obs, legal_actions, self.average_policy)
        action = np.random.choice(len(probs), p=probs)
        return action, probs

    def save(self):
        """Save strategy tables to disk."""
        os.makedirs(self.model_path, exist_ok=True)
        data = {
            "policy": dict(self.policy),
            "average_policy": dict(self.average_policy),
            "regrets": dict(self.regrets),
            "iteration": self.iteration,
            "agent_type": "cfr_plus",
        }
        path = os.path.join(self.model_path, "cfr_plus_model.pkl")
        with open(path, "wb") as f:
            pickle.dump(data, f)
        log.info(f"CFR+ saved: {len(self.regrets)} info sets, {self.iteration} iterations -> {path}")

    def load(self):
        """Load strategy tables from disk. Returns True if successful."""
        path = os.path.join(self.model_path, "cfr_plus_model.pkl")
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        n = self.env.num_actions
        self.policy = collections.defaultdict(
            lambda: np.zeros(n), data.get("policy", {})
        )
        self.average_policy = collections.defaultdict(
            lambda: np.zeros(n), data.get("average_policy", {})
        )
        self.regrets = collections.defaultdict(
            lambda: np.zeros(n), data.get("regrets", {})
        )
        self.iteration = data.get("iteration", 0)
        log.info(f"CFR+ loaded: {len(self.regrets)} info sets, {self.iteration} iterations")
        return True


# ─── External Sampling MCCFR Agent ──────────────────────────────────────────

class MCCFRAgent:
    """
    External Sampling Monte Carlo CFR.

    Instead of traversing all opponent actions at every decision point,
    samples ONE action from the current strategy. This reduces the branching
    factor dramatically:

    - Vanilla CFR:  O(|A|^depth)  per iteration
    - Ext. Sampling: O(|A|^(depth/2))  per iteration  (only hero branches)

    Trade-off: each iteration is cheaper but noisier. You need more iterations,
    but wall-clock time is usually much better for large games.

    Supports optional CFR+ style regret floor via use_plus=True.
    """

    def __init__(self, env, model_path="./models/mccfr", use_plus=True):
        self.use_raw = False
        self.env = env
        self.model_path = model_path
        self.use_plus = use_plus

        self.policy = collections.defaultdict(lambda: np.zeros(env.num_actions))
        self.average_policy = collections.defaultdict(lambda: np.zeros(env.num_actions))
        self.regrets = collections.defaultdict(lambda: np.zeros(env.num_actions))
        self.iteration = 0

    def train(self):
        """One MCCFR iteration with external sampling."""
        self.iteration += 1
        for player_id in range(self.env.num_players):
            self.env.reset()
            self._traverse_external(player_id)
        self._update_policy()

    def _traverse_external(self, player_id):
        """
        External sampling traversal:
        - Hero's turn: try ALL legal actions (full branching)
        - Opponent's turn: sample ONE action from strategy (no branching)
        """
        if self.env.is_over():
            return self.env.get_payoffs()

        current_player = self.env.get_player_id()
        obs, legal_actions = self._get_state(current_player)
        action_probs = self._action_probs(obs, legal_actions, self.policy)

        if current_player == player_id:
            # HERO: traverse all legal actions
            action_utilities = {}
            state_utility = np.zeros(self.env.num_players)

            for action in legal_actions:
                self.env.step(action)
                utility = self._traverse_external(player_id)
                self.env.step_back()
                action_utilities[action] = utility
                state_utility += action_probs[action] * utility

            # Update regrets (no counterfactual prob weighting in external sampling)
            player_state_utility = state_utility[current_player]
            for action in legal_actions:
                regret = action_utilities[action][current_player] - player_state_utility
                if self.use_plus:
                    self.regrets[obs][action] = max(0.0, self.regrets[obs][action] + regret)
                else:
                    self.regrets[obs][action] += regret

            # Update average policy (linear weighting)
            for action in legal_actions:
                self.average_policy[obs][action] += self.iteration * action_probs[action]

            return state_utility

        else:
            # OPPONENT: sample ONE action from current strategy
            legal_probs = np.array([action_probs[a] for a in legal_actions])
            total = legal_probs.sum()
            if total <= 0:
                legal_probs = np.ones(len(legal_actions)) / len(legal_actions)
            else:
                legal_probs /= total

            sampled_action = np.random.choice(legal_actions, p=legal_probs)
            self.env.step(sampled_action)
            result = self._traverse_external(player_id)
            self.env.step_back()
            return result

    def _update_policy(self):
        """Update current policy via regret matching."""
        for obs in self.regrets:
            regret = self.regrets[obs]
            positive = np.maximum(regret, 0)
            total = positive.sum()
            if total > 0:
                self.policy[obs] = positive / total
            else:
                self.policy[obs] = np.ones(self.env.num_actions) / self.env.num_actions

    def _action_probs(self, obs, legal_actions, policy):
        """Get action probabilities, masking illegal actions."""
        if obs not in policy:
            action_probs = np.ones(self.env.num_actions) / self.env.num_actions
            self.policy[obs] = action_probs.copy()
        else:
            action_probs = policy[obs]
        return remove_illegal(action_probs, legal_actions)

    def _get_state(self, player_id):
        """Get (obs_key, legal_actions) for a player."""
        state = self.env.get_state(player_id)
        return state["obs"].tobytes(), list(state["legal_actions"].keys())

    def step(self, state):
        """Select action (for use as opponent in rlcard env)."""
        action, _ = self.eval_step(state)
        return action

    def eval_step(self, state):
        """Select action from average policy. Returns (action, probs_array)."""
        obs = state["obs"].tobytes()
        legal_actions = list(state["legal_actions"].keys())
        probs = self._action_probs(obs, legal_actions, self.average_policy)
        action = np.random.choice(len(probs), p=probs)
        return action, probs

    def save(self):
        """Save strategy tables to disk."""
        os.makedirs(self.model_path, exist_ok=True)
        data = {
            "policy": dict(self.policy),
            "average_policy": dict(self.average_policy),
            "regrets": dict(self.regrets),
            "iteration": self.iteration,
            "use_plus": self.use_plus,
            "agent_type": "mccfr",
        }
        path = os.path.join(self.model_path, "mccfr_model.pkl")
        with open(path, "wb") as f:
            pickle.dump(data, f)
        log.info(f"MCCFR saved: {len(self.regrets)} info sets, {self.iteration} iterations -> {path}")

    def load(self):
        """Load strategy tables from disk. Returns True if successful."""
        path = os.path.join(self.model_path, "mccfr_model.pkl")
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        n = self.env.num_actions
        self.policy = collections.defaultdict(
            lambda: np.zeros(n), data.get("policy", {})
        )
        self.average_policy = collections.defaultdict(
            lambda: np.zeros(n), data.get("average_policy", {})
        )
        self.regrets = collections.defaultdict(
            lambda: np.zeros(n), data.get("regrets", {})
        )
        self.iteration = data.get("iteration", 0)
        self.use_plus = data.get("use_plus", True)
        log.info(f"MCCFR loaded: {len(self.regrets)} info sets, {self.iteration} iterations")
        return True
