import { HandEvaluator } from "./hand-evaluator.ts";

export type EquityResult = {
  winRate: number;
  tieRate: number;
  lossRate: number;
  trials: number;
};

// Precomputed equity lookup for common preflop matchups (hero vs 1 random opponent)
// These avoid running MC trials for known situations, saving ~5-15ms per call
const PREFLOP_EQUITY_TABLE: Record<string, number> = {
  AA: 0.852, KK: 0.824, QQ: 0.800, JJ: 0.775, TT: 0.751,
  "99": 0.723, "88": 0.694, "77": 0.664, "66": 0.635, "55": 0.607,
  "44": 0.579, "33": 0.553, "22": 0.529,
  AKs: 0.674, AQs: 0.660, AJs: 0.651, ATs: 0.643, A9s: 0.631,
  A8s: 0.623, A7s: 0.614, A6s: 0.606, A5s: 0.610, A4s: 0.601,
  A3s: 0.594, A2s: 0.587,
  AKo: 0.654, AQo: 0.638, AJo: 0.628, ATo: 0.619, A9o: 0.605,
  KQs: 0.634, KJs: 0.624, KTs: 0.614, K9s: 0.600, K8s: 0.581,
  KQo: 0.614, KJo: 0.601, KTo: 0.589,
  QJs: 0.612, QTs: 0.601, Q9s: 0.584, QJo: 0.589, QTo: 0.575,
  JTs: 0.591, J9s: 0.572, JTo: 0.566,
  T9s: 0.570, T8s: 0.549, "98s": 0.549, "87s": 0.530, "76s": 0.511,
  "65s": 0.493, "54s": 0.475,
};

export class EquityEstimator {
  private readonly evaluator = new HandEvaluator();
  private readonly ranks = ["2","3","4","5","6","7","8","9","T","J","Q","K","A"];
  private readonly suits = ["s","h","d","c"];

  // LRU cache keyed by "hand|board|opponents" to avoid re-running identical simulations
  private readonly cache = new Map<string, EquityResult>();
  private readonly MAX_CACHE_SIZE = 512;

  public estimateVsRandomRange(
    heroHand: string[],
    board: string[],
    numOpponents: number = 1,
    trials: number = 2000
  ): EquityResult {
    // Fast path: use preflop lookup table when no board cards are present
    if (board.length === 0 && numOpponents === 1) {
      const key = this.handKey(heroHand);
      const tableEquity = PREFLOP_EQUITY_TABLE[key];
      if (tableEquity !== undefined) {
        return {
          winRate: tableEquity,
          tieRate: 0.02,
          lossRate: 1 - tableEquity - 0.02,
          trials: 0,
        };
      }
    }

    // Adaptive trial count: fewer trials for later streets (less variance)
    // and for situations where speed matters more than precision
    const adaptiveTrials = this.getAdaptiveTrials(board, numOpponents, trials);

    const cacheKey = `${heroHand.sort().join(",")}|${board.join(",")}|${numOpponents}|${adaptiveTrials}`;
    const cached = this.cache.get(cacheKey);
    if (cached) return cached;

    const result = this.runMonteCarlo(heroHand, board, numOpponents, adaptiveTrials);

    // Evict oldest entry if cache is full
    if (this.cache.size >= this.MAX_CACHE_SIZE) {
      const firstKey = this.cache.keys().next().value;
      if (firstKey !== undefined) this.cache.delete(firstKey);
    }
    this.cache.set(cacheKey, result);

    return result;
  }

  /**
   * Quick pre-flop equity estimate using lookup table only (zero MC overhead).
   * Returns null if hand not found in table.
   */
  public quickPreflopEquity(heroHand: string[]): number | null {
    const key = this.handKey(heroHand);
    return PREFLOP_EQUITY_TABLE[key] ?? null;
  }

  private getAdaptiveTrials(board: string[], numOpponents: number, requestedTrials: number): number {
    // River: outcome is deterministic given board — fewer trials needed
    if (board.length === 5) return Math.min(requestedTrials, 400);
    // Turn: one card left, moderate variance
    if (board.length === 4) return Math.min(requestedTrials, 600);
    // Flop: two cards left, more variance
    if (board.length === 3) return Math.min(requestedTrials, 800);
    // Preflop fallback (shouldn't be called often due to lookup table)
    return Math.min(requestedTrials, 1000);
  }

  private runMonteCarlo(
    heroHand: string[],
    board: string[],
    numOpponents: number,
    trials: number
  ): EquityResult {
    let wins = 0, ties = 0, losses = 0;
    const dead = new Set([...heroHand, ...board]);
    const deckTemplate = this.buildDeck().filter((c) => !dead.has(c));

    for (let t = 0; t < trials; t++) {
      const deck = [...deckTemplate];
      this.shuffle(deck);

      const villainHands: string[][] = [];
      let idx = 0;
      for (let v = 0; v < numOpponents; v++) {
        villainHands.push([deck[idx], deck[idx + 1]]);
        idx += 2;
      }

      const fullBoard = [...board];
      while (fullBoard.length < 5) {
        fullBoard.push(deck[idx++]);
      }

      const heroEval = this.evaluator.evaluate([...heroHand, ...fullBoard]);
      const bestVillain = Math.max(
        ...villainHands.map((h) => this.evaluator.evaluate([...h, ...fullBoard]).rankValue)
      );

      if (heroEval.rankValue > bestVillain) wins++;
      else if (heroEval.rankValue === bestVillain) ties++;
      else losses++;
    }

    return {
      winRate: wins / trials,
      tieRate: ties / trials,
      lossRate: losses / trials,
      trials,
    };
  }

  private handKey(cards: string[]): string {
    const order = "AKQJT98765432";
    const r1 = cards[0][0].toUpperCase();
    const r2 = cards[1][0].toUpperCase();
    const s1 = cards[0][1];
    const s2 = cards[1][1];
    const sorted = [r1, r2].sort((a, b) => order.indexOf(a) - order.indexOf(b));
    const hi = sorted[0], lo = sorted[1];
    if (hi === lo) return hi + lo;
    return hi + lo + (s1 === s2 ? "s" : "o");
  }

  private buildDeck(): string[] {
    const deck: string[] = [];
    for (const r of this.ranks) {
      for (const s of this.suits) {
        deck.push(`${r}${s}`);
      }
    }
    return deck;
  }

  private shuffle<T>(arr: T[]): void {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
  }
}