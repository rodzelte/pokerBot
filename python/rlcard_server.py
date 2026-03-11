"""
rlcard_server.py — FastAPI bridge between the TypeScript poker bot and rlcard.

Accepts game state as JSON, translates it into rlcard's observation format,
queries a pre-trained DQN agent, and returns a poker action.

Start:
  pip install -r requirements.txt
  python rlcard_server.py          # listens on http://127.0.0.1:5050

Endpoints:
  POST /decide   — main decision endpoint
  GET  /health   — liveness probe
"""
from __future__ import annotations

import os
import json
import logging
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import rlcard
from rlcard.agents import DQNAgent
from rlcard.utils import get_device

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="[rlcard] %(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rlcard_server")

# ── Card conversion tables ───────────────────────────────────────────────────

RANK_MAP = {
    "2": "2", "3": "3", "4": "4", "5": "5", "6": "6",
    "7": "7", "8": "8", "9": "9", "T": "T", "J": "J",
    "Q": "Q", "K": "K", "A": "A",
}

SUIT_MAP = {
    "s": "S", "h": "H", "d": "D", "c": "C",
    "S": "S", "H": "H", "D": "D", "C": "C",
}


def convert_card(card_str: str) -> str:
    """Convert bot card notation (e.g. 'As', 'Td') to rlcard format ('SA', 'DT')."""
    if len(card_str) < 2:
        raise ValueError(f"Invalid card: {card_str}")
    rank = RANK_MAP.get(card_str[0].upper(), card_str[0].upper())
    suit = SUIT_MAP.get(card_str[1], card_str[1].upper())
    # rlcard format: SuitRank  e.g. 'SA' = Ace of Spades
    return f"{suit}{rank}"


# ── rlcard action mapping ───────────────────────────────────────────────────

# rlcard no-limit holdem actions:
#   0 = fold, 1 = check, 2 = call, 3 = raise half pot, 4 = raise full pot, 5 = all-in
RLCARD_ACTION_MAP = {
    0: {"action": "fold",   "size_bb": 0},
    1: {"action": "check",  "size_bb": 0},
    2: {"action": "call",   "size_bb": 0},
    3: {"action": "raise",  "size_bb": 0, "sizing": "half_pot"},
    4: {"action": "raise",  "size_bb": 0, "sizing": "full_pot"},
    5: {"action": "raise",  "size_bb": 0, "sizing": "all_in"},
}


# ── Request / Response models ────────────────────────────────────────────────

class GameState(BaseModel):
    hand: list[str]             # hero hole cards, e.g. ["As", "Kd"]
    board: list[str]            # community cards, e.g. ["Th", "7c", "2s"]
    pot_bb: float               # pot size in big blinds
    hero_stack_bb: float        # hero stack in BBs
    villain_stack_bb: float     # main villain stack in BBs
    facing_bet_bb: float        # current bet hero must call (0 if not facing)
    street: str                 # "preflop" | "flop" | "turn" | "river"
    big_blind: float            # raw chip value of 1 BB


class DecisionResponse(BaseModel):
    action: str                 # "fold" | "check" | "call" | "raise"
    size_bb: float              # bet/raise size in BBs (0 for check/call/fold)
    sizing_type: str            # "half_pot" | "full_pot" | "all_in" | "none"
    raw_action_id: int          # rlcard action id for debugging
    action_probs: dict[str, float]  # probability distribution over actions


# ── Global agent ─────────────────────────────────────────────────────────────

_env = None
_agent = None


def _load_agent():
    """Load rlcard environment and DQN agent."""
    global _env, _agent

    log.info("Loading rlcard no-limit-holdem environment...")
    _env = rlcard.make("no-limit-holdem", config={"game_num_players": 2})

    device = get_device()
    log.info(f"Device: {device}")

    state_shape = _env.state_shape[0]
    action_num = _env.num_actions

    # Always create a fresh agent first
    _agent = DQNAgent(
        num_actions=action_num,
        state_shape=state_shape,
        mlp_layers=[256, 256, 128],
        device=device,
    )

    # Check for a custom pre-trained model checkpoint
    model_path = os.environ.get("RLCARD_MODEL_PATH", "")

    if model_path and os.path.exists(model_path):
        log.info(f"Loading pre-trained model from: {model_path}")
        import torch
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        _agent.q_estimator.qnet.load_state_dict(ckpt["q_net"])
        _agent.target_estimator.qnet.load_state_dict(ckpt["target_net"])
        _agent.total_t = ckpt.get("total_t", 0)
        log.info(f"Model loaded (trained for {_agent.total_t} steps).")
    else:
        log.info("No pre-trained model found — using fresh DQN agent.")
        log.info("For better play, train a model and set RLCARD_MODEL_PATH.")

    log.info("rlcard agent ready.")


# ── Inference ────────────────────────────────────────────────────────────────

def build_observation(state: GameState) -> dict[str, Any]:
    """
    Build a minimal rlcard-style observation from the game state.
    
    rlcard's no-limit-holdem observation has:
      - 'obs': numpy array (feature vector encoding the game state)
      - 'legal_actions': dict mapping action_id -> None
    """
    # Determine legal actions based on game state
    legal = {}
    legal[0] = None  # fold is always legal (unless check is free)
    if state.facing_bet_bb <= 0:
        legal[1] = None  # check
    if state.facing_bet_bb > 0:
        legal[2] = None  # call
    # Can always raise unless all-in
    if state.hero_stack_bb > state.facing_bet_bb:
        legal[3] = None  # raise half pot
        legal[4] = None  # raise pot
        legal[5] = None  # all-in

    # Build feature vector similar to rlcard's encoding
    # rlcard uses a 54-card binary encoding for hand + public cards,
    # plus additional features for pot/stack/action history.
    # We reconstruct a compatible feature vector.
    features = _encode_cards(state.hand, state.board)

    # Append normalised game-state features
    max_stack = max(state.hero_stack_bb, state.villain_stack_bb, 1)
    features.extend([
        state.hero_stack_bb / max_stack,
        state.villain_stack_bb / max_stack,
        state.pot_bb / max_stack,
        state.facing_bet_bb / max_stack,
        _street_to_float(state.street),
    ])

    # Pad or truncate to match env state shape
    target_len = _env.state_shape[0][0] if isinstance(_env.state_shape[0], (list, tuple)) else _env.state_shape[0]
    if len(features) < target_len:
        features.extend([0.0] * (target_len - len(features)))
    features = features[:target_len]

    obs = np.array(features, dtype=np.float32)
    return {"obs": obs, "legal_actions": legal}


def _encode_cards(hand: list[str], board: list[str]) -> list[float]:
    """Binary encoding of cards: 52-dim vector, 1 for each card present."""
    ranks = "23456789TJQKA"
    suits = "SHDC"
    vec = [0.0] * 52
    for card_str in hand + board:
        try:
            rl_card = convert_card(card_str)
            suit_idx = suits.index(rl_card[0])
            rank_idx = ranks.index(rl_card[1])
            vec[suit_idx * 13 + rank_idx] = 1.0
        except (ValueError, IndexError):
            log.warning(f"Could not encode card: {card_str}")
    return vec


def _street_to_float(street: str) -> float:
    return {"preflop": 0.0, "flop": 0.33, "turn": 0.66, "river": 1.0}.get(
        street.lower(), 0.0
    )


def decide(state: GameState) -> DecisionResponse:
    """Run the rlcard agent on the given game state."""
    observation = build_observation(state)

    # rlcard agent.step() expects a state dict with 'obs' and 'legal_actions'
    action_id, action_probs = _agent_step(observation)

    action_info = RLCARD_ACTION_MAP.get(action_id, RLCARD_ACTION_MAP[0])

    # Convert sizing to actual BB amounts
    size_bb = 0.0
    sizing_type = "none"

    if action_info.get("sizing") == "half_pot":
        size_bb = state.pot_bb * 0.5
        sizing_type = "half_pot"
    elif action_info.get("sizing") == "full_pot":
        size_bb = state.pot_bb
        sizing_type = "full_pot"
    elif action_info.get("sizing") == "all_in":
        size_bb = state.hero_stack_bb
        sizing_type = "all_in"

    # Clamp to stack
    size_bb = min(size_bb, state.hero_stack_bb)

    # Build probability labels
    prob_labels = {}
    for aid, prob in enumerate(action_probs):
        label = RLCARD_ACTION_MAP.get(aid, {}).get("action", f"action_{aid}")
        sizing = RLCARD_ACTION_MAP.get(aid, {}).get("sizing", "")
        key = f"{label}_{sizing}" if sizing else label
        prob_labels[key] = round(float(prob), 4)

    return DecisionResponse(
        action=action_info["action"],
        size_bb=round(size_bb, 2),
        sizing_type=sizing_type,
        raw_action_id=action_id,
        action_probs=prob_labels,
    )


def _agent_step(observation: dict) -> tuple[int, list[float]]:
    """Query the DQN agent and return (action_id, probabilities)."""
    # Use eval_step for exploitation (no exploration noise)
    try:
        action_id, probs = _agent.eval_step(observation)
        if isinstance(probs, np.ndarray):
            probs = probs.tolist()
        elif not isinstance(probs, list):
            # Some agents return dict or None
            probs = [0.0] * _env.num_actions
            probs[action_id] = 1.0
        return action_id, probs
    except Exception as e:
        log.error(f"Agent eval_step failed: {e}")
        # Fallback: pick best legal action by heuristic
        legal = observation.get("legal_actions", {})
        # Prefer check > call > fold
        for fallback in [1, 2, 0]:
            if fallback in legal:
                probs = [0.0] * _env.num_actions
                probs[fallback] = 1.0
                return fallback, probs
        return 0, [1.0] + [0.0] * (_env.num_actions - 1)


# ── FastAPI app ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_agent()
    yield

app = FastAPI(title="rlcard-poker-bridge", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "agent_loaded": _agent is not None}


@app.post("/decide", response_model=DecisionResponse)
async def decide_endpoint(state: GameState):
    if _agent is None:
        raise HTTPException(503, "Agent not loaded yet")
    try:
        return decide(state)
    except Exception as e:
        log.exception("Error in /decide")
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("RLCARD_PORT", "5050"))
    log.info(f"Starting rlcard server on port {port}...")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
