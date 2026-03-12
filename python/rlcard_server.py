"""
rlcard_server.py — FastAPI bridge between the TypeScript poker bot and rlcard.

Accepts game state as JSON, translates it into rlcard's observation format,
queries a pre-trained agent (DQN, NFSP, CFR, CFR+, MCCFR, or Deep CFR),
and returns a poker action.

Start:
  pip install -r requirements.txt
  python rlcard_server.py          # listens on http://127.0.0.1:5050

Environment variables:
  RLCARD_AGENT_TYPE   — "dqn" (default), "nfsp", "cfr", "cfr_plus", "mccfr", or "deep_cfr"
  RLCARD_MODEL_PATH   — path to checkpoint file/directory
  RLCARD_DEVICE       — "auto" (default), "cpu", or "cuda"
  RLCARD_CFR_GAME     — game for CFR variants: "limit-holdem" (default), "leduc-holdem"
  RLCARD_PORT         — server port (default 5050)

Endpoints:
  POST /decide   — main decision endpoint
  GET  /health   — liveness probe
  GET  /agent    — current agent info
"""
from __future__ import annotations

import os
import json
import logging
import pickle
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch

import rlcard
from rlcard.agents import DQNAgent, NFSPAgent, CFRAgent

# ── NumPy compat: rlcard 1.1.0 CFRAgent.get_state() calls ndarray.tostring()
# which was removed in NumPy 2.0.  Patch the method to use tobytes() instead.
if not hasattr(np.ndarray(0), 'tostring'):
    _orig_get_state = CFRAgent.get_state
    def _patched_get_state(self, player_id):
        state = self.env.get_state(player_id)
        return state['obs'].tobytes(), list(state['legal_actions'].keys())
    CFRAgent.get_state = _patched_get_state

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
    # Optional: hero's table position for position-aware decisions
    hero_position: str | None = None  # "BTN"|"SB"|"BB"|"UTG"|"MP"|"CO" etc.
    # Optional: request a specific algorithm for this decision
    agent_type: str | None = None
    # Optional: full player info for multi-player awareness
    players: list[dict] | None = None
    # Optional: full action history
    action_history: list[dict] | None = None


class DecisionResponse(BaseModel):
    action: str                 # "fold" | "check" | "call" | "raise"
    size_bb: float              # bet/raise size in BBs (0 for check/call/fold)
    sizing_type: str            # "half_pot" | "full_pot" | "all_in" | "none"
    raw_action_id: int          # rlcard action id for debugging
    action_probs: dict[str, float]  # probability distribution over actions


class SwitchRequest(BaseModel):
    agent_type: str
    model_path: str | None = None


class EnsembleRequest(BaseModel):
    hand: list[str]
    board: list[str]
    pot_bb: float
    hero_stack_bb: float
    villain_stack_bb: float
    facing_bet_bb: float
    street: str
    big_blind: float
    hero_position: str | None = None
    algorithms: list[str]
    weights: list[float] | None = None
    players: list[dict] | None = None
    action_history: list[dict] | None = None


# ── Global agent ─────────────────────────────────────────────────────────────

_env = None
_agent = None
_agent_type = "dqn"  # "dqn" | "nfsp" | "cfr"

# Multi-agent registry for ensemble/cascade/hot-switching
_agent_registry: dict[str, tuple[Any, Any]] = {}  # agent_type -> (agent, env)
_loaded_types: list[str] = []


def resolve_device() -> torch.device:
    """Resolve the torch device for inference from environment settings."""
    device_name = os.environ.get("RLCARD_DEVICE", "auto").lower()

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "RLCARD_DEVICE=cuda was requested but no CUDA-capable GPU is available to PyTorch."
        )

    device = torch.device("cuda:0" if device_name == "cuda" else "cpu")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    return device


def _load_agent():
    """Load rlcard environment and the configured agent."""
    global _env, _agent, _agent_type

    _agent_type = os.environ.get("RLCARD_AGENT_TYPE", "dqn").lower()
    model_path = os.environ.get("RLCARD_MODEL_PATH", "")

    device = resolve_device()
    if device.type == "cuda":
        log.info(f"Device: {device} ({torch.cuda.get_device_name(device)})")
    else:
        log.info(f"Device: {device}")

    if _agent_type == "cfr":
        _load_cfr_agent(model_path)
    elif _agent_type == "cfr_plus":
        _load_cfr_plus_agent(model_path)
    elif _agent_type == "mccfr":
        _load_mccfr_agent(model_path)
    elif _agent_type == "deep_cfr":
        _load_deep_cfr_agent(model_path, device)
    elif _agent_type == "nfsp":
        _load_nfsp_agent(model_path, device)
    else:
        _agent_type = "dqn"
        _load_dqn_agent(model_path, device)

    log.info(f"rlcard agent ready — type: {_agent_type}")
    # Register in multi-agent registry
    _agent_registry[_agent_type] = (_agent, _env)
    if _agent_type not in _loaded_types:
        _loaded_types.append(_agent_type)
    log.info(f"Agent registry: {list(_agent_registry.keys())}")


def _load_dqn_agent(model_path: str, device: torch.device) -> None:
    """Load a DQN agent."""
    global _env, _agent

    log.info("Loading rlcard no-limit-holdem environment (DQN)...")
    _env = rlcard.make("no-limit-holdem", config={"game_num_players": 2})

    _agent = DQNAgent(
        num_actions=_env.num_actions,
        state_shape=_env.state_shape[0],
        mlp_layers=[256, 256, 128],
        device=device,
    )

    if model_path and os.path.exists(model_path):
        log.info(f"Loading pre-trained DQN model from: {model_path}")
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        _agent.q_estimator.qnet.load_state_dict(ckpt["q_net"])
        _agent.target_estimator.qnet.load_state_dict(ckpt["target_net"])
        _agent.total_t = ckpt.get("total_t", 0)
        log.info(f"DQN model loaded (trained for {_agent.total_t} steps).")
    else:
        log.info("No pre-trained DQN model found — using fresh agent.")
        log.info("Train with: python train_dqn.py")


def _load_nfsp_agent(model_path: str, device: torch.device) -> None:
    """Load an NFSP agent."""
    global _env, _agent

    log.info("Loading rlcard no-limit-holdem environment (NFSP)...")
    _env = rlcard.make("no-limit-holdem", config={"game_num_players": 2})

    _agent = NFSPAgent(
        num_actions=_env.num_actions,
        state_shape=_env.state_shape[0],
        hidden_layers_sizes=[256, 256, 128],
        q_mlp_layers=[256, 256, 128],
        device=device,
        evaluate_with="average_policy",
    )

    if model_path and os.path.exists(model_path):
        log.info(f"Loading pre-trained NFSP model from: {model_path}")
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        _agent._rl_agent.q_estimator.qnet.load_state_dict(ckpt["q_net"])
        _agent._rl_agent.target_estimator.qnet.load_state_dict(ckpt["target_net"])
        if "avg_net" in ckpt:
            _agent.policy_network.load_state_dict(ckpt["avg_net"])
        _agent._rl_agent.total_t = ckpt.get("total_t", 0)
        log.info(f"NFSP model loaded (trained for {ckpt.get('episode', '?')} episodes).")
    else:
        log.info("No pre-trained NFSP model found — using fresh agent.")
        log.info("Train with: python train_nfsp.py")


def _load_cfr_agent(model_path: str) -> None:
    """Load a CFR agent."""
    global _env, _agent

    game = os.environ.get("RLCARD_CFR_GAME", "limit-holdem")
    log.info(f"Loading rlcard {game} environment (CFR)...")
    _env = rlcard.make(game, config={"game_num_players": 2})

    _agent = CFRAgent(_env, model_path=model_path or "./models/cfr")

    cfr_pkl = os.path.join(model_path or "./models/cfr", "cfr_model.pkl")
    if os.path.exists(cfr_pkl):
        log.info(f"Loading pre-trained CFR model from: {cfr_pkl}")
        with open(cfr_pkl, "rb") as f:
            data = pickle.load(f)
        if "regrets" in data:
            _agent.regrets = data["regrets"]
        if "average_policy" in data and hasattr(_agent, "average_policy"):
            _agent.average_policy = data["average_policy"]
        if "policy" in data and hasattr(_agent, "policy"):
            _agent.policy = data["policy"]
        if "iteration" in data:
            _agent.iteration = data["iteration"]
        log.info(f"CFR model loaded ({data.get('episode', '?')} iterations, "
                 f"{len(data.get('regrets', {}))} info sets).")
    else:
        log.info("No pre-trained CFR model found — using fresh agent.")
        log.info("Train with: python train_cfr.py")


def _load_cfr_plus_agent(model_path: str) -> None:
    """Load a CFR+ agent via StrategyTable."""
    global _env, _agent
    from strategy_table import StrategyTable

    game = os.environ.get("RLCARD_CFR_GAME", "limit-holdem")
    log.info(f"Loading rlcard {game} environment (CFR+)...")
    _env = rlcard.make(game, config={"game_num_players": 2})

    model_dir = model_path or "./models/cfr_plus"
    try:
        _agent = StrategyTable.load("cfr_plus", model_dir)
        log.info(f"CFR+ strategy loaded: {_agent.summary()}")
    except FileNotFoundError:
        log.info("No pre-trained CFR+ model found — using fresh agent.")
        log.info("Train with: python train_cfr_plus.py")
        from cfr_variants import CFRPlusAgent
        _agent = CFRPlusAgent(_env, model_path=model_dir)


def _load_mccfr_agent(model_path: str) -> None:
    """Load an MCCFR agent via StrategyTable."""
    global _env, _agent
    from strategy_table import StrategyTable

    game = os.environ.get("RLCARD_CFR_GAME", "limit-holdem")
    log.info(f"Loading rlcard {game} environment (MCCFR)...")
    _env = rlcard.make(game, config={"game_num_players": 2})

    model_dir = model_path or "./models/mccfr"
    try:
        _agent = StrategyTable.load("mccfr", model_dir)
        log.info(f"MCCFR strategy loaded: {_agent.summary()}")
    except FileNotFoundError:
        log.info("No pre-trained MCCFR model found — using fresh agent.")
        log.info("Train with: python train_mccfr.py")
        from cfr_variants import MCCFRAgent
        _agent = MCCFRAgent(_env, model_path=model_dir)


def _load_deep_cfr_agent(model_path: str, device: torch.device) -> None:
    """Load a Deep CFR agent."""
    global _env, _agent
    from strategy_table import StrategyTable

    game = os.environ.get("RLCARD_CFR_GAME", "no-limit-holdem")
    log.info(f"Loading rlcard {game} environment (Deep CFR)...")
    _env = rlcard.make(game, config={"game_num_players": 2})

    model_dir = model_path or "./models/deep_cfr"
    device_str = "cuda" if device.type == "cuda" else "cpu"
    try:
        _agent = StrategyTable.load("deep_cfr", model_dir, device=device_str)
        log.info(f"Deep CFR strategy loaded: {_agent.summary()}")
    except FileNotFoundError:
        log.info("No pre-trained Deep CFR model found — using fresh agent.")
        log.info("Train with: python train_deep_cfr.py")
        from deep_cfr_agent import DeepCFRAgent
        _agent = DeepCFRAgent(_env, model_path=model_dir, device=device_str)


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
        _position_to_float(state.hero_position),
    ])

    # Encode per-player positions when available (up to 10 seats)
    if state.players:
        for i in range(10):
            if i < len(state.players):
                p = state.players[i]
                features.append(_position_to_float(p.get("position")))
                features.append(p.get("stack_bb", 0) / max_stack)
                features.append(1.0 if p.get("is_hero") else 0.0)
            else:
                features.extend([0.0, 0.0, 0.0])

    # Encode recent action history positions (last 8 actions)
    if state.action_history:
        recent = state.action_history[-8:]
        for i in range(8):
            if i < len(recent):
                a = recent[i]
                features.append(_action_to_float(a.get("action", "fold")))
                features.append(a.get("amount_bb", 0) / max_stack)
            else:
                features.extend([0.0, 0.0])

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


def _position_to_float(position: str | None) -> float:
    """Encode table position as a 0-1 float (early=0, late=1).

    Late positions (BU, CO) are advantageous in poker because the player
    acts last, gaining more information.  Encoding this lets the model
    learn position-dependent strategy adjustments.
    """
    if not position:
        return 0.5  # unknown → neutral
    _map = {
        "SB": 0.15, "BB": 0.10,
        "UTG": 0.20, "UTG+1": 0.30, "UTG+2": 0.35,
        "MP": 0.45, "MP1": 0.45, "MP2": 0.50,
        "HJ": 0.60, "LJ": 0.55,
        "CO": 0.80, "BU": 1.0, "BTN": 1.0,
    }
    return _map.get(position.upper(), 0.5)


def _action_to_float(action: str) -> float:
    """Encode a poker action as a 0-1 float for the observation vector."""
    _map = {
        "fold": 0.0, "check": 0.2, "call": 0.4,
        "bet": 0.6, "raise": 0.8, "all-in": 1.0,
    }
    return _map.get(action.lower(), 0.0)


def decide(state: GameState) -> DecisionResponse:
    """Run the rlcard agent on the given game state."""
    # Support per-request algorithm selection
    agent_to_use = _agent
    env_to_use = _env
    if state.agent_type and state.agent_type in _agent_registry:
        agent_to_use, env_to_use = _agent_registry[state.agent_type]
        log.info(f"Using per-request agent: {state.agent_type}")

    observation = build_observation(state)

    # rlcard agent.step() expects a state dict with 'obs' and 'legal_actions'
    action_id, action_probs = _agent_step_with(agent_to_use, env_to_use, observation)

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
    """Query the global agent and return (action_id, probabilities)."""
    return _agent_step_with(_agent, _env, observation)


def _agent_step_with(agent: Any, env: Any, observation: dict) -> tuple[int, list[float]]:
    """Query a specific agent and return (action_id, probabilities)."""
    # Use eval_step for exploitation (no exploration noise)
    try:
        action_id, probs = agent.eval_step(observation)
        if isinstance(probs, np.ndarray):
            probs = probs.tolist()
        elif not isinstance(probs, list):
            # Some agents return dict or None
            probs = [0.0] * env.num_actions
            probs[action_id] = 1.0
        return action_id, probs
    except Exception as e:
        log.error(f"Agent eval_step failed: {e}")
        # Fallback: pick best legal action by heuristic
        legal = observation.get("legal_actions", {})
        num_actions = env.num_actions if env else 6
        # Prefer check > call > fold
        for fallback in [1, 2, 0]:
            if fallback in legal:
                probs = [0.0] * num_actions
                probs[fallback] = 1.0
                return fallback, probs
        return 0, [1.0] + [0.0] * (num_actions - 1)


# ── FastAPI app ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_agent()
    yield

app = FastAPI(title="rlcard-poker-bridge", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "agent_loaded": _agent is not None,
        "agent_type": _agent_type,
        "available_agents": _loaded_types,
    }


@app.get("/agent")
async def agent_info():
    """Return info about the currently loaded agent."""
    info = {
        "agent_type": _agent_type,
        "agent_loaded": _agent is not None,
        "supported_types": ["dqn", "nfsp", "cfr", "cfr_plus", "mccfr", "deep_cfr"],
        "available_agents": _loaded_types,
    }
    # Add strategy table info if available
    if hasattr(_agent, "summary"):
        info["strategy_info"] = _agent.summary()
    return info


@app.post("/decide", response_model=DecisionResponse)
async def decide_endpoint(state: GameState):
    if _agent is None:
        raise HTTPException(503, "Agent not loaded yet")
    try:
        return decide(state)
    except Exception as e:
        log.exception("Error in /decide")
        raise HTTPException(500, str(e))


@app.post("/switch")
async def switch_agent(req: SwitchRequest):
    """Hot-switch the active algorithm at runtime."""
    global _env, _agent, _agent_type

    target_type = req.agent_type.lower()
    supported = ["dqn", "nfsp", "cfr", "cfr_plus", "mccfr", "deep_cfr"]
    if target_type not in supported:
        raise HTTPException(400, f"Unsupported agent type: {target_type}. Supported: {supported}")

    # Check if already loaded in registry
    if target_type in _agent_registry:
        _agent, _env = _agent_registry[target_type]
        _agent_type = target_type
        log.info(f"Switched to cached agent: {_agent_type}")
        return {"status": "ok", "agent_type": _agent_type, "cached": True}

    # Load fresh agent
    try:
        model_path = req.model_path or ""
        old_type = _agent_type
        os.environ["RLCARD_AGENT_TYPE"] = target_type
        if model_path:
            os.environ["RLCARD_MODEL_PATH"] = model_path
        _load_agent()
        log.info(f"Hot-switched agent: {old_type} -> {_agent_type}")
        return {"status": "ok", "agent_type": _agent_type, "cached": False}
    except Exception as e:
        log.exception(f"Failed to switch to {target_type}")
        raise HTTPException(500, f"Failed to load {target_type}: {str(e)}")


@app.post("/ensemble")
async def ensemble_endpoint(req: EnsembleRequest):
    """Query multiple algorithms and return weighted ensemble decision."""
    if not _agent_registry:
        raise HTTPException(503, "No agents loaded")

    # Build game state for reuse
    state = GameState(
        hand=req.hand,
        board=req.board,
        pot_bb=req.pot_bb,
        hero_stack_bb=req.hero_stack_bb,
        villain_stack_bb=req.villain_stack_bb,
        facing_bet_bb=req.facing_bet_bb,
        street=req.street,
        big_blind=req.big_blind,
        players=req.players,
        action_history=req.action_history,
    )

    observation = build_observation(state)
    decisions = []
    weights = req.weights or [1.0 / len(req.algorithms)] * len(req.algorithms)

    # Normalise weights
    total_w = sum(weights[:len(req.algorithms)])
    if total_w > 0:
        weights = [w / total_w for w in weights[:len(req.algorithms)]]

    for i, algo in enumerate(req.algorithms):
        if algo not in _agent_registry:
            log.warning(f"Ensemble: agent {algo} not loaded, skipping")
            continue

        agent, env = _agent_registry[algo]
        try:
            action_id, action_probs = _agent_step_with(agent, env, observation)
            action_info = RLCARD_ACTION_MAP.get(action_id, RLCARD_ACTION_MAP[0])

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
            size_bb = min(size_bb, state.hero_stack_bb)

            # Build probability labels
            prob_labels = {}
            for aid, prob in enumerate(action_probs):
                label = RLCARD_ACTION_MAP.get(aid, {}).get("action", f"action_{aid}")
                sizing = RLCARD_ACTION_MAP.get(aid, {}).get("sizing", "")
                key = f"{label}_{sizing}" if sizing else label
                prob_labels[key] = round(float(prob), 4)

            # Compute confidence
            aggressive_prob = sum(
                action_probs[aid] for aid in [3, 4, 5] if aid < len(action_probs)
            )
            call_prob = action_probs[2] if len(action_probs) > 2 else 0
            confidence = min(1.0, aggressive_prob + call_prob * 0.5)

            decisions.append({
                "agent_type": algo,
                "action": action_info["action"],
                "size_bb": round(size_bb, 2),
                "sizing_type": sizing_type,
                "raw_action_id": action_id,
                "action_probs": prob_labels,
                "confidence": round(confidence, 4),
                "weight": weights[i] if i < len(weights) else 0,
            })
        except Exception as e:
            log.warning(f"Ensemble: {algo} failed: {e}")

    if not decisions:
        raise HTTPException(500, "All ensemble agents failed")

    # Weighted voting: aggregate action probabilities
    combined_probs: dict[str, float] = {}
    for d in decisions:
        w = d["weight"]
        for key, prob in d["action_probs"].items():
            combined_probs[key] = combined_probs.get(key, 0) + prob * w

    # Pick best action from combined probabilities
    # Map prob keys back to action IDs
    best_key = max(combined_probs, key=lambda k: combined_probs[k])
    best_action = best_key.split("_")[0]  # "raise_half_pot" -> "raise"
    best_sizing_type = "none"
    best_size_bb = 0.0

    if "half_pot" in best_key:
        best_size_bb = state.pot_bb * 0.5
        best_sizing_type = "half_pot"
    elif "full_pot" in best_key:
        best_size_bb = state.pot_bb
        best_sizing_type = "full_pot"
    elif "all_in" in best_key:
        best_size_bb = state.hero_stack_bb
        best_sizing_type = "all_in"
    best_size_bb = min(best_size_bb, state.hero_stack_bb)

    combined_confidence = sum(d["confidence"] * d["weight"] for d in decisions)

    return {
        "decisions": decisions,
        "combined": {
            "action": best_action,
            "size_bb": round(best_size_bb, 2),
            "sizing_type": best_sizing_type,
            "confidence": round(combined_confidence, 4),
            "action_probs": {k: round(v, 4) for k, v in combined_probs.items()},
        },
    }


@app.get("/strategy")
async def strategy_info():
    """Return strategy storage information for all loaded agents."""
    info = {}
    for agent_type, (agent, env) in _agent_registry.items():
        entry: dict[str, Any] = {"loaded": True}
        if hasattr(agent, "summary"):
            entry["summary"] = agent.summary()
        if hasattr(agent, "regrets"):
            entry["info_sets"] = len(agent.regrets)
        if hasattr(agent, "total_t"):
            entry["training_steps"] = agent.total_t
        if hasattr(agent, "iteration"):
            entry["iterations"] = agent.iteration
        info[agent_type] = entry
    return info


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("RLCARD_PORT", "5050"))
    log.info(f"Starting rlcard server on port {port}...")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
