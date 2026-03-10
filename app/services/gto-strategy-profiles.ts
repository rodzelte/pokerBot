/**
 * gto-strategy-profiles.ts
 *
 * Defines strategy profiles that the engine selects from based on game state,
 * villain tendencies, and table dynamics. Each profile controls:
 *   - Opening / defending / 3-bet ranges (tight vs. loose)
 *   - Bet sizing frequencies and sizes
 *   - Bluff inclusion rates in betting ranges
 *   - Mixed-strategy frequencies (for GTO randomisation)
 *
 * GTO note: A pure Nash-equilibrium solver requires game-tree traversal that is
 * too expensive for a real-time bot. Instead we approximate GTO by:
 *   1. Using solver-derived range thresholds (equity/EV cutoffs)
 *   2. Randomising indifferent spots via a frequency table
 *   3. Adapting those frequencies based on villain stats (exploitative overlay)
 */

export type StyleProfile = "gto" | "tight" | "loose" | "lag";

/** Per-street sizing options */
export type SizingScheme = {
  smallFraction: number;   // e.g. 0.33
  medFraction: number;     // e.g. 0.50
  largeFraction: number;   // e.g. 0.75
  overbetFraction: number; // e.g. 1.20
};

/** Bluff / mixed-strategy frequency table */
export type MixedFrequencies = {
  /** How often to bluff-bet air on flop when checked to (0–1) */
  flopCBetAir: number;
  /** How often to bluff-bet air on turn when checked to (0–1) */
  turnCBetAir: number;
  /** How often to bluff-bet air on river when checked to (0–1) */
  riverBluff: number;
  /** How often to semi-bluff a gutshot (weaker draw) */
  gutshotSemiBluff: number;
  /** How often to 3-bet light (suited connectors / broadways vs. late-pos openers) */
  lightThreeBet: number;
  /** How often to check-raise a strong-made hand for deception */
  checkRaiseStrong: number;
  /** How often to slowplay a monster (check instead of bet) */
  slowplayMonster: number;
};

export type StrategyProfile = {
  name: StyleProfile;
  description: string;

  /** Equity threshold (vs. random range) below which we fold facing a bet */
  foldEquityThreshold: number;

  /** Equity threshold to call a flop/turn/river bet */
  callEquityThreshold: number;

  /** Minimum equity to semi-bluff (has a draw, not yet made) */
  semiBluffMinEquity: number;

  /** Minimum equity to value-bet (polarised approach) */
  valueBetMinEquity: number;

  /** Minimum pot-odds-adjusted equity multiplier for draws (implied odds factor) */
  impliedOddsFactor: number;

  sizing: SizingScheme;
  mixed: MixedFrequencies;

  /**
   * Bankroll risk fraction (Kelly-inspired).
   * Controls how aggressively to size up when ahead.
   * 0.25 = quarter Kelly (conservative), 0.5 = half Kelly, 1.0 = full Kelly (aggressive)
   */
  kellyFraction: number;
};

// ─── Profile Definitions ────────────────────────────────────────────────────

export const GTO_PROFILE: StrategyProfile = {
  name: "gto",
  description: "Balanced GTO-approximation — unexploitable, mixed strategies",
  foldEquityThreshold: 0.30,
  callEquityThreshold: 0.35,
  semiBluffMinEquity: 0.28,
  valueBetMinEquity: 0.58,
  impliedOddsFactor: 0.80, // 80% of pot-odds required (implied upside)
  sizing: {
    smallFraction: 0.33,
    medFraction: 0.55,
    largeFraction: 0.75,
    overbetFraction: 1.25,
  },
  mixed: {
    flopCBetAir: 0.33,      // bluff c-bet air 33% (GTO range-bet approach)
    turnCBetAir: 0.20,
    riverBluff: 0.25,
    gutshotSemiBluff: 0.45,
    lightThreeBet: 0.18,
    checkRaiseStrong: 0.30, // mix checking strong hands 30% for balance
    slowplayMonster: 0.20,
  },
  kellyFraction: 0.35,
};

export const TIGHT_PROFILE: StrategyProfile = {
  name: "tight",
  description: "Tight-Aggressive (TAG) — value-heavy, minimal bluffs",
  foldEquityThreshold: 0.38,
  callEquityThreshold: 0.42,
  semiBluffMinEquity: 0.38,
  valueBetMinEquity: 0.62,
  impliedOddsFactor: 0.90, // tighter implied-odds requirement
  sizing: {
    smallFraction: 0.40,
    medFraction: 0.60,
    largeFraction: 0.80,
    overbetFraction: 1.00, // rarely overbet
  },
  mixed: {
    flopCBetAir: 0.15,
    turnCBetAir: 0.08,
    riverBluff: 0.10,
    gutshotSemiBluff: 0.25,
    lightThreeBet: 0.08,
    checkRaiseStrong: 0.15,
    slowplayMonster: 0.10,
  },
  kellyFraction: 0.25,
};

export const LOOSE_PROFILE: StrategyProfile = {
  name: "loose",
  description: "Loose-Aggressive (LAG) — wider ranges, more pressure",
  foldEquityThreshold: 0.22,
  callEquityThreshold: 0.28,
  semiBluffMinEquity: 0.22,
  valueBetMinEquity: 0.52,
  impliedOddsFactor: 0.65,
  sizing: {
    smallFraction: 0.28,
    medFraction: 0.50,
    largeFraction: 0.80,
    overbetFraction: 1.30,
  },
  mixed: {
    flopCBetAir: 0.52,
    turnCBetAir: 0.35,
    riverBluff: 0.38,
    gutshotSemiBluff: 0.60,
    lightThreeBet: 0.30,
    checkRaiseStrong: 0.40,
    slowplayMonster: 0.15,
  },
  kellyFraction: 0.50,
};

export const LAG_PROFILE: StrategyProfile = {
  name: "lag",
  description: "Max-pressure LAG — exploits passive villains aggressively",
  foldEquityThreshold: 0.18,
  callEquityThreshold: 0.24,
  semiBluffMinEquity: 0.18,
  valueBetMinEquity: 0.48,
  impliedOddsFactor: 0.55,
  sizing: {
    smallFraction: 0.25,
    medFraction: 0.55,
    largeFraction: 0.90,
    overbetFraction: 1.50,
  },
  mixed: {
    flopCBetAir: 0.65,
    turnCBetAir: 0.48,
    riverBluff: 0.45,
    gutshotSemiBluff: 0.70,
    lightThreeBet: 0.40,
    checkRaiseStrong: 0.50,
    slowplayMonster: 0.10,
  },
  kellyFraction: 0.60,
};

const ALL_PROFILES: Record<StyleProfile, StrategyProfile> = {
  gto: GTO_PROFILE,
  tight: TIGHT_PROFILE,
  loose: LOOSE_PROFILE,
  lag: LAG_PROFILE,
};

export function getProfile(name: StyleProfile): StrategyProfile {
  return ALL_PROFILES[name];
}

/**
 * Select the most profitable profile given villain stats and stack depth.
 *
 * Logic:
 *  - vs. unknowns / tight villains → GTO (least exploitable)
 *  - vs. passive fish (high VPIP, low PFR) → LAG (maximise value extraction)
 *  - vs. maniacs (high VPIP + high PFR) → TIGHT (let them bluff into us)
 *  - short-stacked → TIGHT (no room for fancy plays)
 */
export function selectProfile(
  villainVPIP: number,
  villainPFR: number,
  villainSampleSize: number,
  effectiveStackBB: number
): StrategyProfile {
  // Not enough data — default to GTO
  if (villainSampleSize < 15) return GTO_PROFILE;

  // Very short stacks — tighten up, no complex multi-street bluffs
  if (effectiveStackBB < 20) return TIGHT_PROFILE;

  const isPassiveFish = villainVPIP > 45 && villainPFR < 15;
  const isManiac      = villainVPIP > 50 && villainPFR > 35;
  const isLAG         = villainVPIP > 35 && villainPFR > 25;
  const isTAG         = villainVPIP < 22 && villainPFR > 14;

  if (isPassiveFish) return LAG_PROFILE;   // Hammer passive fish
  if (isManiac)      return TIGHT_PROFILE; // Let them spew, trap them
  if (isLAG)         return GTO_PROFILE;   // GTO vs. tough regs
  if (isTAG)         return LOOSE_PROFILE; // Apply pressure on tight players

  return GTO_PROFILE; // Default balanced
}

/**
 * Randomise a decision based on a mixed-strategy frequency.
 * Returns true if the "active" branch should fire.
 *
 * Uses a seeded-style deterministic hash on hand+street so decisions
 * are consistent within a hand (same hand key → same random branch),
 * preventing easy exploitation via observation of timing.
 */
export function mixedStrategyFires(
  frequency: number,
  seed: string
): boolean {
  if (frequency <= 0) return false;
  if (frequency >= 1) return true;
  // Deterministic hash: consistent per hand so we don't flip mid-hand
  let hash = 0;
  for (let i = 0; i < seed.length; i++) {
    hash = (Math.imul(31, hash) + seed.charCodeAt(i)) >>> 0;
  }
  const normalised = (hash % 1000) / 1000;
  return normalised < frequency;
}