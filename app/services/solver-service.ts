/**
 * solver-service.ts
 *
 * Lightweight poker hand evaluation & equity estimation service
 * built on npm packages (no external solver binary needed).
 *
 * Packages used:
 *   - pokersolver      -- Hand evaluation & comparison (up to 7 cards)
 *   - poker-odds-calc  -- Monte-Carlo equity / odds computation
 *   - phe              -- (already in project) fast hand evaluation
 */

// @ts-ignore -- pokersolver has no type declarations
import pokerSolver from "pokersolver";

// --- Types -----------------------------------------------------------------

export interface SolverConfig {
  enabled: boolean;
  monteCarloTrials: number;
  useOnStreets: string[];
  maxSolveTimeMs: number;
}

export interface SolverInput {
  heroHand: string[];
  board: string[];
  potBB: number;
  effectiveStackBB: number;
  isHeroIP: boolean;
  heroRange: string;
  villainRange: string;
  street: "flop" | "turn" | "river";
}

export interface SolverAction {
  action: "fold" | "check" | "call" | "bet" | "raise" | "all-in";
  sizeBB: number;
  frequency: number;
  ev: number;
}

export interface SolverResult {
  actions: SolverAction[];
  bestAction: SolverAction;
  source: "monte-carlo";
  solveTimeMs: number;
  equity: number;
}

// --- Hand Evaluation Helpers -----------------------------------------------

export function evaluateHand(cards: string[]): { rank: number; name: string; descr: string } {
  const hand = pokerSolver.Hand.solve(cards);
  return { rank: hand.rank as number, name: hand.name as string, descr: hand.descr as string };
}

export function compareHands(a: string[], b: string[]): number {
  const ha = pokerSolver.Hand.solve(a);
  const hb = pokerSolver.Hand.solve(b);
  const winners = pokerSolver.Hand.winners([ha, hb]);
  if (winners.length === 2) return 0;
  return winners[0] === ha ? 1 : -1;
}

// --- Range Helpers ---------------------------------------------------------

export function rangeToSolverString(
  hands: Set<string>,
  weights?: Map<string, number>
): string {
  const parts: string[] = [];
  for (const h of hands) {
    const w = weights?.get(h) ?? 1;
    if (w >= 0.99) parts.push(h);
    else parts.push(`${h}:${w.toFixed(2)}`);
  }
  return parts.join(",");
}

export function getDefaultHeroRange(position: string): string {
  const ranges: Record<string, string> = {
    UTG:     "AA,KK,QQ,JJ,TT,99,AKs,AQs,AJs,KQs,AKo,AQo",
    "UTG+1": "AA,KK,QQ,JJ,TT,99,88,AKs,AQs,AJs,KQs,AKo,AQo,AJo",
    MP:      "AA,KK,QQ,JJ,TT,99,88,77,AKs,AQs,AJs,ATs,KQs,QJs,JTs,AKo,AQo,AJo,KQo",
    LJ:      "AA,KK,QQ,JJ,TT,99,88,77,66,AKs,AQs,AJs,ATs,KQs,KJs,QJs,JTs,T9s,AKo,AQo,AJo,KQo",
    HJ:      "AA,KK,QQ,JJ,TT,99,88,77,66,55,AKs,AQs,AJs,ATs,A9s,KQs,KJs,QJs,JTs,T9s,98s,AKo,AQo,AJo,KQo,QJo",
    CO:      "AA,KK,QQ,JJ,TT,99,88,77,66,55,44,AKs,AQs,AJs,ATs,A9s,A8s,KQs,KJs,KTs,QJs,QTs,JTs,T9s,98s,87s,AKo,AQo,AJo,ATo,KQo,KJo,QJo",
    BU:      "AA,KK,QQ,JJ,TT,99,88,77,66,55,44,33,22,AKs,AQs,AJs,ATs,A9s,A8s,A7s,A6s,A5s,A4s,A3s,A2s,KQs,KJs,KTs,K9s,QJs,QTs,Q9s,JTs,J9s,T9s,98s,87s,76s,AKo,AQo,AJo,ATo,KQo,KJo,QJo,JTo",
    SB:      "AA,KK,QQ,JJ,TT,99,88,77,66,55,AKs,AQs,AJs,ATs,A9s,KQs,KJs,KTs,QJs,QTs,JTs,T9s,98s,87s,AKo,AQo,AJo,ATo,KQo,KJo",
    BB:      "AA,KK,QQ,JJ,TT,99,88,77,66,55,44,33,22,AKs,AQs,AJs,ATs,A9s,A8s,A7s,A6s,A5s,A4s,A3s,A2s,KQs,KJs,KTs,K9s,K8s,QJs,QTs,Q9s,JTs,J9s,T9s,T8s,98s,97s,87s,86s,76s,75s,65s,54s,AKo,AQo,AJo,ATo,A9o,KQo,KJo,KTo,QJo,QTo,JTo",
  };
  return ranges[position] ?? ranges["CO"];
}

// --- Normalisation ---------------------------------------------------------

function normaliseCard(c: string): string {
  let s = c.trim();
  if (s.startsWith("10")) s = "T" + s.slice(2);
  return s[0].toUpperCase() + s.slice(1).toLowerCase();
}

// --- Monte-Carlo Equity ----------------------------------------------------

const MC_RANKS = ["2","3","4","5","6","7","8","9","T","J","Q","K","A"];
const MC_SUITS = ["s","h","d","c"];

function buildDeck(): string[] {
  const deck: string[] = [];
  for (const r of MC_RANKS) for (const s of MC_SUITS) deck.push(`${r}${s}`);
  return deck;
}

function shuffle(arr: string[]): void {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}

function computeEquityMC(
  heroHand: string[],
  board: string[],
  trials: number = 1000,
): { equity: number; win: number; tie: number; loss: number } {
  const dead = new Set([...heroHand.map(normaliseCard), ...board.map(normaliseCard)]);
  const deckTemplate = buildDeck().filter((c) => !dead.has(c));
  let wins = 0, ties = 0;

  for (let t = 0; t < trials; t++) {
    const deck = [...deckTemplate];
    shuffle(deck);
    let idx = 0;
    const villainHand = [deck[idx++], deck[idx++]];
    const fullBoard = [...board.map(normaliseCard)];
    while (fullBoard.length < 5) fullBoard.push(deck[idx++]);

    const heroCards = [...heroHand.map(normaliseCard), ...fullBoard];
    const villainCards = [...villainHand, ...fullBoard];
    const cmp = compareHands(heroCards, villainCards);
    if (cmp > 0) wins++;
    else if (cmp === 0) ties++;
  }

  const winPct = wins / trials;
  const tiePct = ties / trials;
  return {
    equity: winPct + tiePct * 0.5,
    win: winPct,
    tie: tiePct,
    loss: 1 - winPct - tiePct,
  };
}

// --- Unified Solver Service ------------------------------------------------

export class SolverService {
  private cache = new Map<string, SolverResult>();
  private config: SolverConfig | null = null;
  private enabled = false;

  async init(config: SolverConfig): Promise<void> {
    this.config = config;
    this.enabled = config.enabled;
    if (this.enabled) {
      console.log("[Solver] Monte-Carlo equity solver ready (pokersolver + poker-odds-calc).");
    } else {
      console.log("[Solver] Solver integration disabled.");
    }
  }

  isAvailable(): boolean {
    return this.enabled;
  }

  shouldSolve(street: string): boolean {
    if (!this.config?.enabled) return false;
    return this.config.useOnStreets.includes(street);
  }

  async solve(input: SolverInput): Promise<SolverResult | null> {
    if (!this.enabled) return null;

    const key = this.getCacheKey(input);
    const cached = this.cache.get(key);
    if (cached) {
      console.log("[Solver] Cache hit.");
      return cached;
    }

    const start = performance.now();
    const mc = computeEquityMC(input.heroHand, input.board);
    const elapsed = performance.now() - start;
    const equity = mc.equity;

    const actions: SolverAction[] = [];

    if (equity >= 0.70) {
      const sizeBB = Math.min(input.potBB * 0.75, input.effectiveStackBB);
      actions.push({ action: "bet",   sizeBB, frequency: 0.7, ev: equity * input.potBB });
      actions.push({ action: "raise", sizeBB, frequency: 0.2, ev: equity * input.potBB });
      actions.push({ action: "call",  sizeBB: 0, frequency: 0.1, ev: equity * input.potBB * 0.5 });
    } else if (equity >= 0.50) {
      const sizeBB = Math.min(input.potBB * 0.5, input.effectiveStackBB);
      actions.push({ action: "call",  sizeBB: 0, frequency: 0.5, ev: equity * input.potBB * 0.6 });
      actions.push({ action: "bet",   sizeBB, frequency: 0.35, ev: equity * input.potBB * 0.5 });
      actions.push({ action: "check", sizeBB: 0, frequency: 0.15, ev: equity * input.potBB * 0.3 });
    } else if (equity >= 0.35) {
      actions.push({ action: "check", sizeBB: 0, frequency: 0.5, ev: equity * input.potBB * 0.3 });
      actions.push({ action: "call",  sizeBB: 0, frequency: 0.35, ev: equity * input.potBB * 0.2 });
      actions.push({ action: "fold",  sizeBB: 0, frequency: 0.15, ev: 0 });
    } else {
      actions.push({ action: "fold",  sizeBB: 0, frequency: 0.7, ev: 0 });
      actions.push({ action: "check", sizeBB: 0, frequency: 0.2, ev: 0 });
      const bluffSize = Math.min(input.potBB * 0.5, input.effectiveStackBB);
      actions.push({ action: "bet",   sizeBB: bluffSize, frequency: 0.1, ev: -bluffSize * 0.3 });
    }

    const best = actions.reduce((a, b) => b.ev > a.ev ? b : a);
    const solverResult: SolverResult = {
      actions, bestAction: best, source: "monte-carlo", solveTimeMs: elapsed, equity,
    };

    this.cache.set(key, solverResult);
    console.log(
      `[Solver] MC equity: ${(equity * 100).toFixed(1)}% in ${elapsed.toFixed(0)}ms -> ` +
      `best: ${best.action}` +
      (best.sizeBB > 0 ? ` ${best.sizeBB.toFixed(1)}bb` : "") +
      ` (freq: ${(best.frequency * 100).toFixed(0)}%)`
    );
    return solverResult;
  }

  clearCache(): void {
    this.cache.clear();
  }

  private getCacheKey(input: SolverInput): string {
    return [
      input.heroHand.join(""),
      input.board.join(""),
      input.potBB.toFixed(0),
      input.effectiveStackBB.toFixed(0),
      input.isHeroIP ? "IP" : "OOP",
    ].join("|");
  }
}
