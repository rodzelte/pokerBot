/**
 * equity-estimator.ts
 *
 * Hybrid approach:
 *   - `pokersolver` (via HandEvaluator) for hand evaluation in MC simulations
 *   - `poker-odds-calc` for known hero-vs-villain equity (quickEquityAgainstHand)
 *   - Preflop lookup table for instant preflop equity
 *
 * Same exported types & API as the original estimator.
 */

import { HandEvaluator } from "./hand-evaluator.ts";
// @ts-ignore — poker-odds-calc ships JS only
import { TexasHoldem as OddsTable } from "poker-odds-calc";

export type EquityMethod = "preflop-table" | "poker-odds-calc" | "monte-carlo" | "hand-evaluator";

export type EquityResult = {
  winRate: number;
  tieRate: number;
  lossRate: number;
  trials: number;
};

// Precomputed equity lookup for common preflop matchups (hero vs 1 random opponent)
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

/** Normalise a card like "10h" → "Th", "kd" → "Kd" */
function normaliseCard(c: string): string {
  let s = c.trim();
  if (s.startsWith("10")) s = "T" + s.slice(2);
  return s[0].toUpperCase() + s.slice(1).toLowerCase();
}

export class EquityEstimator {
  private readonly evaluator = new HandEvaluator();
  private readonly ranks = ["2","3","4","5","6","7","8","9","T","J","Q","K","A"];
  private readonly suits = ["s","h","d","c"];

  private readonly cache = new Map<string, EquityResult>();
  private readonly MAX_CACHE_SIZE = 512;
  private lastMethod: EquityMethod = "monte-carlo";

  /** Returns which calculation method was used for the last equity call. */
  public getLastMethod(): EquityMethod { return this.lastMethod; }

  /**
   * Estimate hero equity vs N random opponents via Monte-Carlo simulation.
   * Uses pokersolver-backed HandEvaluator for each trial.
   */
  public estimateVsRandomRange(
    heroHand: string[],
    board: string[],
    numOpponents: number = 1,
    trials: number = 2000
  ): EquityResult {
    const hand = heroHand.map(normaliseCard);
    const brd  = board.map(normaliseCard);

    // Fast path: preflop lookup table
    if (brd.length === 0 && numOpponents === 1) {
      const key = this.handKey(hand);
      const tableEquity = PREFLOP_EQUITY_TABLE[key];
      if (tableEquity !== undefined) {
        this.lastMethod = "preflop-table";
        return {
          winRate: tableEquity,
          tieRate: 0.02,
          lossRate: 1 - tableEquity - 0.02,
          trials: 0,
        };
      }
    }

    const adaptiveTrials = this.getAdaptiveTrials(brd, trials);
    const cacheKey = `${[...hand].sort().join(",")}|${brd.join(",")}|${numOpponents}|${adaptiveTrials}`;
    const cached = this.cache.get(cacheKey);
    if (cached) return cached;

    this.lastMethod = "monte-carlo";
    const result = this.runMonteCarlo(hand, brd, numOpponents, adaptiveTrials);

    if (this.cache.size >= this.MAX_CACHE_SIZE) {
      const firstKey = this.cache.keys().next().value;
      if (firstKey !== undefined) this.cache.delete(firstKey);
    }
    this.cache.set(cacheKey, result);
    return result;
  }

  /** Quick pre-flop equity estimate using lookup table only. */
  public quickPreflopEquity(heroHand: string[]): number | null {
    const key = this.handKey(heroHand);
    return PREFLOP_EQUITY_TABLE[key] ?? null;
  }

  /**
   * Compute hero win equity against a specific villain hand.
   * Uses poker-odds-calc when board is incomplete, HandEvaluator when complete.
   */
  public quickEquityAgainstHand(
    heroHand: string[],
    villainHand: string[],
    board: string[],
    _trials: number = 200
  ): number {
    const hero = heroHand.map(normaliseCard);
    const villain = villainHand.map(normaliseCard);
    const brd = board.map(normaliseCard);

    // Check for card collisions before evaluating
    const allCards = [...hero, ...villain, ...brd];
    if (new Set(allCards).size !== allCards.length) {
      // Cards overlap — cannot compute, return 0.5 (coin flip)
      return 0.5;
    }

    // River: deterministic single evaluation via pokersolver
    if (brd.length === 5) {
      this.lastMethod = "hand-evaluator";
      const hEval = this.evaluator.evaluate([...hero, ...brd]);
      const vEval = this.evaluator.evaluate([...villain, ...brd]);
      if (hEval.rankValue > vEval.rankValue) return 1;
      if (hEval.rankValue === vEval.rankValue) return 0.5;
      return 0;
    }

    // Use poker-odds-calc for incomplete boards (flop/turn)
    try {
      const table = new OddsTable();
      table.addPlayer(hero as any);
      table.addPlayer(villain as any);
      if (brd.length > 0) {
        table.setBoard(brd as any);
      }
      const res = table.calculate();
      const heroResult = res.getPlayers()[0];
      const winPct = parseFloat(heroResult.getWinsPercentageString()) / 100;
      const tiePct = parseFloat(heroResult.getTiesPercentageString()) / 100;
      this.lastMethod = "poker-odds-calc";
      return winPct + tiePct * 0.5;
    } catch {
      // Fallback: use pokersolver-based MC if poker-odds-calc fails
      this.lastMethod = "monte-carlo";
      return this.fallbackEquityAgainstHand(hero, villain, brd);
    }
  }

  // ── Internal ──────────────────────────────────────────────────────────────

  private getAdaptiveTrials(board: string[], requestedTrials: number): number {
    if (board.length === 5) return Math.min(requestedTrials, 400);
    if (board.length === 4) return Math.min(requestedTrials, 600);
    if (board.length === 3) return Math.min(requestedTrials, 800);
    return Math.min(requestedTrials, 1000);
  }

  private runMonteCarlo(
    heroHand: string[], board: string[], numOpponents: number, trials: number
  ): EquityResult {
    let wins = 0, ties = 0, losses = 0;
    const dead = new Set([...heroHand.map(normaliseCard), ...board.map(normaliseCard)]);
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
      while (fullBoard.length < 5) fullBoard.push(deck[idx++]);

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

  private fallbackEquityAgainstHand(
    heroHand: string[], villainHand: string[], board: string[]
  ): number {
    const ranks = ["2","3","4","5","6","7","8","9","T","J","Q","K","A"];
    const suits = ["s","h","d","c"];
    const dead = new Set([
      ...heroHand.map(normaliseCard),
      ...villainHand.map(normaliseCard),
      ...board.map(normaliseCard),
    ]);
    const deck: string[] = [];
    for (const r of ranks) for (const s of suits) {
      const c = `${r}${s}`;
      if (!dead.has(c)) deck.push(c);
    }

    let wins = 0;
    const trials = 200;
    for (let t = 0; t < trials; t++) {
      const d = [...deck];
      for (let i = d.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [d[i], d[j]] = [d[j], d[i]];
      }
      const fullBoard = [...board];
      let idx = 0;
      while (fullBoard.length < 5) fullBoard.push(d[idx++]);
      const hEval = this.evaluator.evaluate([...heroHand, ...fullBoard]);
      const vEval = this.evaluator.evaluate([...villainHand, ...fullBoard]);
      if (hEval.rankValue > vEval.rankValue) wins += 1;
      else if (hEval.rankValue === vEval.rankValue) wins += 0.5;
    }
    return wins / trials;
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
    for (const r of this.ranks) for (const s of this.suits) deck.push(`${r}${s}`);
    return deck;
  }

  private shuffle<T>(arr: T[]): void {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
  }
}