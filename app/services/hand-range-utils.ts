/**
 * hand-range-utils.ts
 *
 * Utilities for:
 *  - Constructing and querying preflop hand ranges
 *  - Estimating range advantage on a given board texture
 *  - Blocker effects (do our hole cards block villain's strong holdings?)
 *  - Combo counting for balanced bluff:value ratios
 *
 * Ranges are expressed as arrays of hand keys ("AKs", "QJo", "TT", etc.)
 * and converted to combo counts when needed for frequency calculations.
 */

// ─── Types ────────────────────────────────────────────────────────────────

export type HandKey = string; // e.g. "AKs", "QJo", "TT"

export type RangeStats = {
  totalCombos: number;
  valueCombos: number;
  bluffCombos: number;
  bluffToValueRatio: number;
  /** Recommended value: ~1:2 bluff-to-value on river, ~1:1.5 on turn */
  isBalanced: boolean;
};

export type BoardTexture = {
  isPaired: boolean;
  isMonotone: boolean;
  isTwoTone: boolean;
  isRainbow: boolean;
  hasHighCard: boolean;  // top board card >= J
  connected: boolean;   // has 2 or more connecting cards
  wetScore: number;     // 0–4 composite wetness score
};

// ─── Combo Counting ────────────────────────────────────────────────────────

/**
 * Number of distinct two-card combinations for a hand key.
 *  Pairs:    C(4,2) = 6
 *  Suited:   4
 *  Offsuit:  12
 */
export function combosForHandKey(key: HandKey): number {
  if (key.length === 2) return 6;                         // pocket pair
  if (key.endsWith("s")) return 4;                        // suited
  if (key.endsWith("o")) return 12;                       // offsuit
  return 6;
}

/**
 * Total combos in a range.
 */
export function countRangeCombos(range: HandKey[]): number {
  return range.reduce((sum, h) => sum + combosForHandKey(h), 0);
}

// ─── Board Texture Analysis ───────────────────────────────────────────────

export function analyzeBoardTexture(board: string[]): BoardTexture {
  if (board.length < 3) {
    return {
      isPaired: false, isMonotone: false, isTwoTone: false,
      isRainbow: true, hasHighCard: false, connected: false, wetScore: 0,
    };
  }

  const ranks = board.map((c) => c[0].toUpperCase());
  const suits = board.map((c) => c[1].toLowerCase());
  const rankOrder = "AKQJT98765432";

  // Paired board
  const rankCounts = new Map<string, number>();
  for (const r of ranks) rankCounts.set(r, (rankCounts.get(r) ?? 0) + 1);
  const isPaired = [...rankCounts.values()].some((v) => v >= 2);

  // Suit distribution
  const suitCounts = new Map<string, number>();
  for (const s of suits) suitCounts.set(s, (suitCounts.get(s) ?? 0) + 1);
  const maxSuit = Math.max(...suitCounts.values());
  const isMonotone = maxSuit >= 3;
  const isTwoTone  = maxSuit === 2 && board.length === 3;
  const isRainbow  = maxSuit === 1;

  // High-card presence (J or better on board)
  const highCards = new Set(["A","K","Q","J"]);
  const hasHighCard = ranks.some((r) => highCards.has(r));

  // Connectedness — count pairs of cards within 3 ranks
  const idxs = ranks
    .map((r) => rankOrder.indexOf(r))
    .filter((i) => i >= 0)
    .sort((a, b) => a - b);
  let connected = false;
  for (let i = 0; i < idxs.length - 1; i++) {
    if (idxs[i + 1] - idxs[i] <= 2) { connected = true; break; }
  }

  // Wet score 0–4
  let wetScore = 0;
  if (isMonotone) wetScore += 2;
  else if (isTwoTone) wetScore += 1;
  if (connected) wetScore += 1;
  if (!isPaired && !hasHighCard) wetScore += 1; // low connected boards hit draws more

  return { isPaired, isMonotone, isTwoTone, isRainbow, hasHighCard, connected, wetScore };
}

// ─── Range Advantage ─────────────────────────────────────────────────────

/**
 * Estimate hero's range advantage on this board vs. villain's range.
 *
 * Returns a score from -1 (villain has huge range advantage) to +1 (hero does).
 * Used to calibrate c-bet frequency: higher advantage → c-bet more.
 *
 * Simplified heuristic (a real solver does full range × range × board evaluation).
 */
export function estimateRangeAdvantage(
  heroPosition: string,
  villainPosition: string,
  board: string[],
  texture: BoardTexture
): number {
  let score = 0;

  const inPositionBonus = isInPosition(heroPosition, villainPosition) ? 0.15 : -0.10;
  score += inPositionBonus;

  // Preflop aggressor (BTN/CO/HJ) has range advantage on most boards
  const aggressorPositions = new Set(["BU","CO","HJ","SB"]);
  if (aggressorPositions.has(heroPosition)) score += 0.20;

  // High boards (A/K/Q high) favour the preflop raiser's range
  if (texture.hasHighCard) score += 0.15;

  // Paired boards neutralise range advantage
  if (texture.isPaired) score -= 0.10;

  // Very wet boards help the caller's wide range more
  if (texture.wetScore >= 3) score -= 0.15;

  return Math.max(-1, Math.min(1, score));
}

function isInPosition(hero: string, villain: string): boolean {
  const order = ["UTG","UTG+1","MP","LJ","HJ","CO","BU","SB","BB"];
  return order.indexOf(hero) > order.indexOf(villain);
}

// ─── Blocker Detection ────────────────────────────────────────────────────

/**
 * Does the hero's hand block villain's nut combos?
 * Returns a blocker score 0–3 (higher = more blocking power).
 *
 * Examples:
 *  - Holding Ad on Kd-Td-5d blocks nut flush
 *  - Holding Ah blocks AA and AK combos
 *  - Holding Kh blocks KK and AK combos
 */
export function calcBlockerScore(hand: string[], board: string[]): number {
  const ranks = hand.map((c) => c[0].toUpperCase());
  const suits = hand.map((c) => c[1].toLowerCase());
  const boardSuits = board.map((c) => c[1].toLowerCase());
  const boardRanks = board.map((c) => c[0].toUpperCase());

  let score = 0;

  // Block nut flush: hold a card of the suit that appears most on board
  const suitCounts = new Map<string, number>();
  for (const s of boardSuits) suitCounts.set(s, (suitCounts.get(s) ?? 0) + 1);
  const dominantSuit = [...suitCounts.entries()].sort((a,b) => b[1]-a[1])[0];
  if (dominantSuit && dominantSuit[1] >= 2 && suits.includes(dominantSuit[0])) {
    score += 1;
  }

  // Block top-pair and overpair combos
  const topBoardRank = boardRanks.sort((a,b) =>
    "AKQJT98765432".indexOf(a) - "AKQJT98765432".indexOf(b)
  )[0];
  if (ranks.includes(topBoardRank ?? "")) score += 1;

  // Block nut straight: hold an A or the card completing the most likely straight
  if (ranks.includes("A")) score += 0.5;

  return Math.min(3, score);
}

// ─── Bluff : Value Ratio Analysis ────────────────────────────────────────

/**
 * Given a range partition into value and bluff hands, assess balance.
 * GTO river ratio ≈ 1 bluff : 2 value (pot-sized bet).
 * GTO turn ratio ≈ 1 bluff : 1.5 value.
 * GTO flop ratio ≈ 1 bluff : 1 value (range-betting).
 */
export function assessRangeBalance(
  valueCombos: number,
  bluffCombos: number,
  street: string
): RangeStats {
  const total = valueCombos + bluffCombos;
  const ratio = valueCombos > 0 ? bluffCombos / valueCombos : 999;

  const targetRatios: Record<string, number> = {
    flop: 1.0,
    turn: 0.67,
    river: 0.50,
    preflop: 0.33,
  };
  const target = targetRatios[street] ?? 0.50;
  const isBalanced = Math.abs(ratio - target) < 0.25;

  return {
    totalCombos: total,
    valueCombos,
    bluffCombos,
    bluffToValueRatio: ratio,
    isBalanced,
  };
}

// ─── Pot Odds & EV ────────────────────────────────────────────────────────

/**
 * Minimum defence frequency (MDF) — the fraction of our range we must
 * continue with to prevent villain from profitably bluffing any two cards.
 *   MDF = pot / (pot + bet)
 */
export function minimumDefenceFrequency(potBB: number, betBB: number): number {
  if (betBB <= 0) return 1;
  return potBB / (potBB + betBB);
}

/**
 * Required equity to call a bet (simple pot-odds call threshold).
 */
export function potOddsEquity(potBB: number, betBB: number): number {
  if (betBB <= 0) return 0;
  return betBB / (potBB + betBB);
}

/**
 * Expected Value of a bluff given fold equity.
 *   EV = foldEq * pot - (1 - foldEq) * betSize
 */
export function bluffEV(potBB: number, betBB: number, foldEquity: number): number {
  return foldEquity * potBB - (1 - foldEquity) * betBB;
}

/**
 * Estimated fold equity based on villain VPIP/PFR (tighter villains fold more).
 * Returns a value 0–1.
 */
export function estimateFoldEquity(
  villainVPIP: number,
  villainPFR: number,
  sampleHands: number,
  betSizeFraction: number,
  boardTexture: BoardTexture
): number {
  if (sampleHands < 10) return 0.45; // unknown villain, assume average

  // Base: tighter villains fold more
  let base = 0.65 - (villainVPIP / 200); // 45% base for 40vpip, 55% for 20vpip

  // Aggression-based adjustment: aggressive villains call/raise more
  const aggrFactor = Math.min(1, villainPFR / 30);
  base -= aggrFactor * 0.12;

  // Large bets get more folds
  base += (betSizeFraction - 0.5) * 0.10;

  // Wet boards: villain more likely to have draws → calls more
  if (boardTexture.wetScore >= 3) base -= 0.08;

  // Paired boards: fewer made-hand combos → calls slightly less
  if (boardTexture.isPaired) base += 0.05;

  return Math.max(0.15, Math.min(0.80, base));
}

// ─── Hand Strength Tiers ──────────────────────────────────────────────────

/**
 * Categorise a normalised hand key into a rough preflop strength tier.
 * Used to decide whether to use value or bluff frequency for mixed spots.
 */
export type PreflopTier = "premium" | "strong" | "medium" | "speculative" | "marginal";

const TIER_MAP: Record<string, PreflopTier> = {
  AA:"premium", KK:"premium", QQ:"premium", JJ:"strong",
  TT:"strong",  AKs:"premium", AKo:"strong", AQs:"strong",
  AQo:"strong", AJs:"strong", KQs:"strong",
  "99":"medium", "88":"medium", ATs:"medium", AJo:"medium",
  KJs:"medium", KQo:"medium", QJs:"medium", JTs:"medium",
  "77":"medium", "66":"speculative", "55":"speculative",
  "44":"speculative", "33":"speculative", "22":"speculative",
  A9s:"speculative", A8s:"speculative", A7s:"speculative",
  A6s:"speculative", A5s:"speculative", A4s:"speculative",
  A3s:"speculative", A2s:"speculative",
  KTs:"speculative", K9s:"speculative",
  QTs:"speculative", Q9s:"speculative",
  JTs:"speculative", J9s:"speculative",
  T9s:"speculative", "98s":"speculative", "87s":"speculative",
  "76s":"speculative", "65s":"speculative", "54s":"speculative",
  ATo:"marginal", A9o:"marginal", KJo:"marginal", KTo:"marginal",
  QJo:"marginal", QTo:"marginal", JTo:"marginal",
};

export function getPreflopTier(handKey: HandKey): PreflopTier {
  return TIER_MAP[handKey] ?? "marginal";
}