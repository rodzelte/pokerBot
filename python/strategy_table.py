"""
strategy_table.py — Unified strategy lookup for all CFR-trained poker agents.

Converts any trained CFR variant (vanilla, CFR+, MCCFR, Deep CFR) into a
fast lookup interface for serving poker decisions.

For tabular agents (CFR, CFR+, MCCFR):
  Direct dictionary lookup: state → action probabilities.
  Lightning fast for known states; falls back to uniform for unseen states.

For Deep CFR:
  Neural network forward pass → softmax → action probabilities.
  Generalises to ANY state, even ones never seen during training.

Usage:
  # Load from trained model
  table = StrategyTable.load("cfr_plus", "./models/cfr_plus")
  action, probs = table.eval_step(state)

  # Check coverage
  print(table.summary())
"""
from __future__ import annotations

import os
import pickle
import logging

import numpy as np
import torch

log = logging.getLogger("strategy_table")


class StrategyTable:
    """
    Unified strategy lookup for all CFR variants.

    Provides a single eval_step(state) interface regardless of whether the
    underlying strategy is a tabular dictionary or a neural network.
    Compatible with rlcard's agent interface and the FastAPI server.
    """

    def __init__(self):
        self.agent_type: str = ""
        self.num_actions: int = 0
        self.info_sets: int = 0
        self.iterations: int = 0
        self._lookup: dict | None = None      # tabular: obs_bytes → probs array
        self._network: torch.nn.Module | None = None  # Deep CFR strategy net
        self._device: torch.device = torch.device("cpu")
        self.use_raw = False

    @classmethod
    def load(cls, agent_type: str, model_path: str, device: str = "cpu") -> "StrategyTable":
        """
        Load a strategy table from a trained model directory.

        Args:
            agent_type: "cfr", "cfr_plus", "mccfr", or "deep_cfr"
            model_path: directory containing the model files
            device: "cpu" or "cuda"
        """
        table = cls()
        table.agent_type = agent_type
        table._device = torch.device(device)

        if agent_type in ("cfr", "cfr_plus", "mccfr"):
            table._load_tabular(model_path, agent_type)
        elif agent_type == "deep_cfr":
            table._load_deep_cfr(model_path)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        return table

    def _load_tabular(self, model_path: str, agent_type: str):
        """Load a tabular strategy from pickle."""
        filenames = {
            "cfr": "cfr_model.pkl",
            "cfr_plus": "cfr_plus_model.pkl",
            "mccfr": "mccfr_model.pkl",
        }
        pkl_path = os.path.join(model_path, filenames[agent_type])
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Model not found: {pkl_path}")

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        self._lookup = data.get("average_policy", {})
        self.iterations = data.get("iteration", 0)
        self.info_sets = len(self._lookup)

        # Infer num_actions from any stored policy vector
        if self._lookup:
            sample = next(iter(self._lookup.values()))
            self.num_actions = len(sample)
        else:
            self.num_actions = 0

        log.info(
            f"Loaded {agent_type} strategy table: "
            f"{self.info_sets} info sets, {self.iterations} iterations"
        )

    def _load_deep_cfr(self, model_path: str):
        """Load a Deep CFR strategy network."""
        from deep_cfr_agent import StrategyNetwork

        pt_path = os.path.join(model_path, "deep_cfr_model.pt")
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"Model not found: {pt_path}")

        data = torch.load(pt_path, map_location=self._device, weights_only=False)
        self.num_actions = data["num_actions"]
        state_dim = data["state_dim"]
        self.iterations = data.get("iteration", 0)

        self._network = StrategyNetwork(state_dim, self.num_actions)
        self._network.load_state_dict(data["strategy_net"])
        self._network.to(self._device)
        self._network.eval()

        params = sum(p.numel() for p in self._network.parameters())
        log.info(
            f"Loaded deep_cfr strategy network: "
            f"{params:,} params, {self.iterations} iterations"
        )

    def get_action_probs(self, state: dict) -> np.ndarray:
        """
        Get action probability distribution for a game state.

        Args:
            state: dict with 'obs' (numpy array) and 'legal_actions' (dict)

        Returns:
            numpy array of shape (num_actions,) with probabilities
        """
        legal_actions = list(state["legal_actions"].keys())
        if not legal_actions:
            return np.zeros(self.num_actions)

        if self._lookup is not None:
            return self._lookup_probs(state, legal_actions)
        elif self._network is not None:
            return self._network_probs(state, legal_actions)
        else:
            raise RuntimeError("No strategy loaded")

    def _lookup_probs(self, state: dict, legal_actions: list[int]) -> np.ndarray:
        """Tabular lookup with fallback to uniform."""
        obs_key = state["obs"].tobytes()

        if obs_key in self._lookup:
            probs = self._lookup[obs_key].copy()
            # Zero out illegal actions and renormalise
            for i in range(len(probs)):
                if i not in legal_actions:
                    probs[i] = 0.0
            total = probs.sum()
            if total > 0:
                probs /= total
                return probs

        # Unknown state: uniform over legal actions
        probs = np.zeros(self.num_actions)
        for a in legal_actions:
            probs[a] = 1.0 / len(legal_actions)
        return probs

    def _network_probs(self, state: dict, legal_actions: list[int]) -> np.ndarray:
        """Neural network inference with masked softmax."""
        obs = state["obs"].flatten().astype(np.float32)

        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self._device)
            logits = self._network(obs_t).cpu().numpy().flatten()

        probs = np.zeros(self.num_actions, dtype=np.float64)
        legal_logits = np.array([logits[a] for a in legal_actions], dtype=np.float64)
        legal_logits -= legal_logits.max()
        exp_logits = np.exp(legal_logits)
        legal_probs = exp_logits / (exp_logits.sum() + 1e-8)

        for i, a in enumerate(legal_actions):
            probs[a] = legal_probs[i]

        return probs

    def step(self, state):
        """Select action (for use as opponent in rlcard env)."""
        action, _ = self.eval_step(state)
        return action

    def eval_step(self, state: dict) -> tuple[int, np.ndarray]:
        """
        rlcard-compatible eval: sample action from strategy.

        Returns:
            (action_id, probs_array)
        """
        probs = self.get_action_probs(state)
        if probs.sum() <= 0:
            # Safety fallback
            legal_actions = list(state["legal_actions"].keys())
            if legal_actions:
                return legal_actions[0], probs
            return 0, probs
        action = np.random.choice(len(probs), p=probs)
        return action, probs

    def summary(self) -> str:
        """Human-readable summary of the loaded strategy."""
        if self._lookup is not None:
            return (
                f"StrategyTable({self.agent_type}, "
                f"{self.info_sets:,} info sets, "
                f"{self.iterations} iterations)"
            )
        elif self._network is not None:
            params = sum(p.numel() for p in self._network.parameters())
            return (
                f"StrategyTable({self.agent_type}, "
                f"{params:,} network params, "
                f"{self.iterations} iterations)"
            )
        return f"StrategyTable({self.agent_type}, empty)"

    @staticmethod
    def export_from_cfr_agent(agent, agent_type: str, export_path: str):
        """
        Export a trained rlcard CFRAgent (or variant) to a standalone
        strategy table file. This creates a pickle with just the average_policy,
        usable for fast inference without the full agent machinery.
        """
        avg_policy = {}
        if hasattr(agent, "average_policy"):
            avg_policy = dict(agent.average_policy)
        elif hasattr(agent, "policy"):
            avg_policy = dict(agent.policy)

        data = {
            "average_policy": avg_policy,
            "iteration": getattr(agent, "iteration", 0),
            "agent_type": agent_type,
            "info_sets": len(avg_policy),
        }

        os.makedirs(os.path.dirname(export_path) or ".", exist_ok=True)
        with open(export_path, "wb") as f:
            pickle.dump(data, f)

        log.info(
            f"Exported {agent_type} strategy: "
            f"{len(avg_policy)} info sets -> {export_path}"
        )
