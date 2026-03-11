// ─── opponent-range.ts ──────────────────────────────────────────────────────

import { EquityEstimator, EquityResult } from "./equity-estimator.ts";
import { HandEvaluator } from "./hand-evaluator.ts";

export type OpponentRange = {
  hands: Set<string>;         // Possible hole cards
  weightedHands: Map<string, number>; // Probability weight for each hand
  aggressionCount: number;
  vpip: number;               // % voluntarily put in pot
  pfr: number;                // % preflop raise
};

export class OpponentModel {
  private equityEstimator = new EquityEstimator();
  private handEvaluator   = new HandEvaluator();
  private allHands        = new Set<string>([
    "AA","KK","QQ","JJ","TT","99","88","77","66","55","44","33","22",
    "AKs","AQs","AJs","ATs","KQs","KJs","QJs","JTs","T9s","98s",
    "AKo","AQo","AJo","KQo"
  ]);

  public createInitialRange(vpip: number = 40, pfr: number = 25): OpponentRange {
    const hands = new Set(this.allHands);
    const weightedHands = new Map<string, number>();
    for (const h of hands) weightedHands.set(h, 1); // uniform
    return { hands, weightedHands, aggressionCount: 0, vpip, pfr };
  }

  public updatePreflopRange(range: OpponentRange, action: "fold"|"call"|"raise", position: string) {
    const newHands = new Set<string>();
    for (const h of range.hands) {
      if (action === "fold" && !this.isFoldableHand(h, position)) continue;
      if (action === "raise" && !this.isRaiseableHand(h, position)) continue;
      if (action === "call" && !this.isCallableHand(h, position)) continue;
      newHands.add(h);
    }
    range.hands = newHands;
    this.updateWeights(range, action);
  }

  public updatePostflopRange(range: OpponentRange, action: "check"|"bet"|"raise", board: string[], street: "flop"|"turn"|"river") {
    const newHands = new Set<string>();
    const boardSet = new Set(board.map(c => c[0].toUpperCase() + c.slice(1).toLowerCase()));
    for (const h of range.hands) {
      const cards = this.handToCardsAvoidingBoard(h, boardSet);
      if (!cards) continue; // no valid suit combo exists — skip this hand
      const madeHand = this.handEvaluator.evaluate([...cards, ...board]);
      if (action === "check" && this.isLikelyWeakHand(madeHand, street)) newHands.add(h);
      if ((action === "bet" || action === "raise") && this.isLikelyStrongHand(madeHand, street)) newHands.add(h);
    }
    range.hands = newHands;
    this.updateWeights(range, action);
  }

  public computeEquityVsRange(hand: string[], range: OpponentRange, board: string[]): number {
    let equitySum = 0;
    let totalWeight = 0;
    const dead = new Set([
      ...hand.map(c => c[0].toUpperCase() + c.slice(1).toLowerCase()),
      ...board.map(c => c[0].toUpperCase() + c.slice(1).toLowerCase()),
    ]);
    for (const h of range.hands) {
      const cards = this.handToCardsAvoidingBoard(h, dead);
      if (!cards) continue; // conflicts with hero hand or board
      const weight = range.weightedHands.get(h) ?? 1;
      const equity = this.equityEstimator.quickEquityAgainstHand(hand, cards, board) ?? 0;
      equitySum += equity * weight;
      totalWeight += weight;
    }
    return totalWeight > 0 ? equitySum / totalWeight : 0;
  }

  // ── Helpers ───────────────────────────────────────────────────────────────

  private handToCards(hand: string): string[] {
    // e.g. "AKs" → ["As","Ks"]
    if (hand.length === 2) return [hand[0]+"h", hand[1]+"s"]; // assume suited
    const [r1,r2,t] = hand.split("") as [string,string,string];
    return t === "s" ? [r1+"h", r2+"h"] : [r1+"h", r2+"s"];
  }

  /**
   * Convert a hand key (e.g. "AKs") to two concrete cards, picking suits
   * that don't collide with any cards in `deadCards`.
   * Returns null if no valid suit assignment exists.
   */
  private handToCardsAvoidingBoard(hand: string, deadCards: Set<string>): string[] | null {
    const suits = ["h","d","c","s"];
    const r1 = hand[0];
    const r2 = hand.length === 2 ? hand[1] : hand[1]; // pair or two ranks
    const isSuited = hand.endsWith("s");
    const isPair   = hand.length === 2;

    if (isSuited) {
      // Both cards share a suit — find a suit where neither card is dead
      for (const s of suits) {
        const c1 = r1 + s, c2 = r2 + s;
        if (!deadCards.has(c1) && !deadCards.has(c2)) return [c1, c2];
      }
      return null;
    }

    if (isPair) {
      // Pocket pair — find two different suits where neither card is dead
      for (let i = 0; i < suits.length; i++) {
        for (let j = i + 1; j < suits.length; j++) {
          const c1 = r1 + suits[i], c2 = r2 + suits[j];
          if (!deadCards.has(c1) && !deadCards.has(c2)) return [c1, c2];
        }
      }
      return null;
    }

    // Offsuit — two different suits
    for (const s1 of suits) {
      for (const s2 of suits) {
        if (s1 === s2) continue;
        const c1 = r1 + s1, c2 = r2 + s2;
        if (!deadCards.has(c1) && !deadCards.has(c2)) return [c1, c2];
      }
    }
    return null;
  }

  private isFoldableHand(hand: string, position: string) {
    // crude: premium hands never fold
    return !["AA","KK","QQ","AKs","AKo"].includes(hand);
  }

  private isRaiseableHand(hand: string, position: string) {
    return ["AA","KK","QQ","JJ","AKs","AKo","AQs","AQo"].includes(hand);
  }

  private isCallableHand(hand: string, position: string) {
    return true; // any hand could call
  }

  private isLikelyWeakHand(madeHand: any, street: string) {
    return !madeHand || madeHand.category === "pair" || madeHand.category === "two-pair";
  }

  private isLikelyStrongHand(madeHand: any, street: string) {
    if (!madeHand) return false;
    return ["three-of-a-kind","straight","flush","full-house","four-of-a-kind"].includes(madeHand.category);
  }

  private updateWeights(range: OpponentRange, action: string) {
    const factor = action === "raise" ? 1.5 : action === "fold" ? 0.1 : 1.0;
    for (const h of range.hands) {
      const w = range.weightedHands.get(h) ?? 1;
      range.weightedHands.set(h, w * factor);
    }
  }
}
