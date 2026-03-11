/**
 * hand-evaluator.ts
 *
 * Powered by the `pokersolver` npm package.
 * Same exported types & API as the original hand-written evaluator.
 */

// @ts-ignore — pokersolver ships without type declarations
import pokerSolver from "pokersolver";

export type HandCategory =
  | "high-card"
  | "pair"
  | "two-pair"
  | "three-of-a-kind"
  | "straight"
  | "flush"
  | "full-house"
  | "four-of-a-kind"
  | "straight-flush";

export type EvaluatedHand = {
  category: HandCategory;
  rankValue: number;
  bestFive: string[];
};

/** Map pokersolver's `.name` to our canonical HandCategory strings. */
const NAME_TO_CATEGORY: Record<string, HandCategory> = {
  "Straight Flush":  "straight-flush",
  "Four of a Kind":  "four-of-a-kind",
  "Full House":      "full-house",
  "Flush":           "flush",
  "Straight":        "straight",
  "Three of a Kind": "three-of-a-kind",
  "Two Pair":        "two-pair",
  "Pair":            "pair",
  "High Card":       "high-card",
};

export class HandEvaluator {
  /**
   * Evaluate the best 5-card hand out of 5-7 cards.
   * Returns category, numeric rank (higher = better), and the best 5 cards.
   */
  public evaluate(cards: string[]): EvaluatedHand {
    if (cards.length < 5 || cards.length > 7) {
      throw new Error(`Expected 5 to 7 cards, got ${cards.length}`);
    }

    const solved = pokerSolver.Hand.solve(cards);
    const category: HandCategory = NAME_TO_CATEGORY[solved.name as string] ?? "high-card";
    const bestFive: string[] = (solved.cards ?? [])
      .slice(0, 5)
      .map((c: any) => c.value + c.suit);

    return {
      category,
      rankValue: solved.rank as number,
      bestFive: bestFive.length === 5 ? bestFive : cards.slice(0, 5),
    };
  }
}