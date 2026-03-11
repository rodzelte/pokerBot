/**
 * range-advantage.ts
 *
 * Approximates solver-style range advantage using
 * preflop ranges + board interaction scoring.
 */

export type RangeAdvantageResult = {
  heroAdvantage: number;
  nutAdvantage: number;
};

const PREMIUM_RANKS = ["A", "K", "Q", "J", "T"];

/**
 * Score how well a board interacts with a range.
 */
function scoreRangeInteraction(range: string[], board: string[]): number {
  let score = 0;

  for (const hand of range) {
    const r1 = hand[0];
    const r2 = hand[1];

    if (board.some(c => c[0] === r1)) score += 2;
    if (board.some(c => c[0] === r2)) score += 2;

    if (PREMIUM_RANKS.includes(r1)) score += 1;
    if (PREMIUM_RANKS.includes(r2)) score += 1;
  }

  return score / Math.max(range.length, 1);
}

/**
 * Estimate range vs range advantage
 */
export function calculateRangeAdvantage(
  heroRange: string[],
  villainRange: string[],
  board: string[]
): RangeAdvantageResult {

  const heroScore = scoreRangeInteraction(heroRange, board);
  const villainScore = scoreRangeInteraction(villainRange, board);

  const heroAdvantage = heroScore - villainScore;

  const nutAdvantage =
    heroRange.filter(h => h.includes("A") || h.includes("K")).length -
    villainRange.filter(h => h.includes("A") || h.includes("K")).length;

  return {
    heroAdvantage,
    nutAdvantage
  };
}
