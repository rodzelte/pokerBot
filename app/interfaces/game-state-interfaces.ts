/**
 * game-state-interfaces.ts
 *
 * Unified game state snapshot and algorithm-specific parameter interfaces.
 * These types define the complete data contract between bot.ts and all
 * AI algorithm modules (CFR, DQN, NFSP, Local Engine, LLM).
 */

// ── Enums ───────────────────────────────────────────────────────────────────

export type AlgorithmType = "cfr" | "cfr_plus" | "mccfr" | "deep_cfr" | "dqn" | "nfsp";

export type PokerStreet = "preflop" | "flop" | "turn" | "river";

export type PokerActionType = "fold" | "check" | "call" | "bet" | "raise" | "all-in";

export type SelectionMode = "single" | "ensemble" | "cascade";

// ── Player Info ─────────────────────────────────────────────────────────────

export interface PlayerInfo {
  playerId: string;
  name: string;
  seat: number;
  position: string;          // "SB" | "BB" | "UTG" | "MP" | ... | "CO" | "BU"
  stackChips: number;        // stack in raw chip value
  stackBB: number;           // stack in big blinds
  isHero: boolean;
  isDealer: boolean;
  /** Villain stats (only populated if available from memory) */
  stats?: {
    totalHands: number;
    vpip: number;             // 0-100
    pfr: number;              // 0-100
    aggression?: number;      // 0-100
    threeBetPct?: number;     // 0-100
    cBetPct?: number;         // 0-100
    foldTo3BetPct?: number;   // 0-100
  };
}

// ── Action Record ───────────────────────────────────────────────────────────

export interface ActionRecord {
  playerId: string;
  position: string;
  action: PokerActionType;
  amountChips: number;
  amountBB: number;
  street: PokerStreet;
}

// ── Game Rules & Parameters ─────────────────────────────────────────────────

export interface GameRules {
  gameType: string;           // "NLH" (No Limit Hold'em)
  bigBlind: number;           // raw chip value
  smallBlind: number;         // raw chip value
  maxPlayers: number;
  ante?: number;
}

// ── Unified Game State Snapshot ─────────────────────────────────────────────

/**
 * Complete snapshot of the game state at a decision point.
 * This is the single source of truth passed from bot.ts → algorithm modules.
 */
export interface GameStateSnapshot {
  /** Game identification */
  gameId: string;
  handId: string;

  /** Game rules */
  rules: GameRules;

  /** Current street */
  street: PokerStreet;

  /** Hero's private cards, e.g. ["As", "Kd"] */
  heroHand: string[];

  /** Community cards, e.g. ["Th", "7c", "2s"] */
  board: string[];

  /** All players at the table (including hero) */
  players: PlayerInfo[];

  /** Hero player reference */
  hero: PlayerInfo;

  /** Current pot size in big blinds */
  potBB: number;

  /** Current pot size in raw chips */
  potChips: number;

  /** Amount hero needs to call (0 if check is free) */
  facingBetBB: number;

  /** Full action history for this hand, grouped by street */
  actionHistory: ActionRecord[];

  /** Number of raises on the current street */
  raisesThisStreet: number;

  /** Players still in the pot */
  playersInPot: number;

  /** Effective stack (min of hero + shortest relevant villain) in BB */
  effectiveStackBB: number;

  /** Stack-to-Pot Ratio */
  spr: number;

  /** Timestamp of snapshot */
  timestamp: number;
}

// ── Algorithm-Specific Parameters ───────────────────────────────────────────

/**
 * Parameters specific to CFR-family algorithms.
 * CFR needs reach probabilities and information set keys.
 */
export interface CFRParams {
  algorithmType: "cfr" | "cfr_plus" | "mccfr" | "deep_cfr";
  /** Game environment for CFR (limit-holdem for tabular, no-limit-holdem for deep) */
  cfrGame: string;
  /** Number of training iterations completed */
  trainedIterations?: number;
  /** Path to strategy/model files */
  modelPath: string;
}

/**
 * Parameters specific to DQN algorithm.
 */
export interface DQNParams {
  algorithmType: "dqn";
  /** Exploration rate (0 = fully greedy, 1 = fully random) */
  epsilon: number;
  /** Hidden layer sizes for the Q-network */
  mlpLayers: number[];
  /** Path to checkpoint */
  modelPath: string;
  /** Device for inference */
  device: string;
}

/**
 * Parameters specific to NFSP algorithm.
 */
export interface NFSPParams {
  algorithmType: "nfsp";
  /** Anticipatory parameter (η) — probability of using RL vs supervised policy */
  anticipatoryParam: number;
  /** Hidden layer sizes */
  hiddenLayers: number[];
  /** Q-network hidden layers */
  qMlpLayers: number[];
  /** Evaluation mode: "average_policy" or "best_response" */
  evaluateWith: string;
  /** Path to checkpoint */
  modelPath: string;
  /** Device for inference */
  device: string;
}

export type AlgorithmParams = CFRParams | DQNParams | NFSPParams;

// ── Algorithm Decision Output ───────────────────────────────────────────────

export interface AlgorithmDecision {
  /** Which algorithm made this decision */
  source: AlgorithmType;
  /** The recommended action */
  action: PokerActionType;
  /** Bet/raise size in big blinds (0 for check/call/fold) */
  sizeBB: number;
  /** Confidence score 0-1 */
  confidence: number;
  /** Action probability distribution */
  actionProbs: Record<string, number>;
  /** Sizing type label (e.g. "half_pot", "full_pot", "all_in") */
  sizingType: string;
  /** Raw action ID from the algorithm */
  rawActionId?: number;
}

// ── Ensemble Configuration ──────────────────────────────────────────────────

export interface EnsembleConfig {
  /** How to select/combine algorithm outputs */
  mode: SelectionMode;
  /** Algorithms to use (in priority order for cascade mode) */
  algorithms: AlgorithmType[];
  /** Weights for ensemble mode (same order as algorithms) */
  weights?: number[];
  /** Minimum confidence threshold to accept a decision (cascade mode) */
  minConfidence?: number;
}

// ── Strategy Storage ────────────────────────────────────────────────────────

export interface StrategyStorageInfo {
  algorithmType: AlgorithmType;
  /** Path to stored strategy/weights */
  storagePath: string;
  /** Last updated timestamp */
  lastUpdated?: number;
  /** Number of info sets (CFR) or training steps (DQN/NFSP) */
  size?: number;
  /** Additional metadata */
  metadata?: Record<string, unknown>;
}
