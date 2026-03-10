import { HandEvaluator } from "./hand-evaluator.ts";

export type EquityResult = {
  winRate: number;
  tieRate: number;
  lossRate: number;
  trials: number;
};

export class EquityEstimator {
  private readonly evaluator = new HandEvaluator();
  private readonly ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"];
  private readonly suits = ["s", "h", "d", "c"];

  public estimateVsRandomRange(
    heroHand: string[],
    board: string[],
    numOpponents: number = 1,
    trials: number = 2000
  ): EquityResult {
    let wins = 0;
    let ties = 0;
    let losses = 0;

    const dead = new Set([...heroHand, ...board]);

    for (let t = 0; t < trials; t++) {
      const deck = this.buildDeck().filter((c) => !dead.has(c));
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
      const villainEvals = villainHands.map((h) =>
        this.evaluator.evaluate([...h, ...fullBoard])
      );

      const bestVillain = Math.max(...villainEvals.map((v) => v.rankValue));

      if (heroEval.rankValue > bestVillain) {
        wins++;
      } else if (heroEval.rankValue === bestVillain) {
        ties++;
      } else {
        losses++;
      }
    }

    return {
      winRate: wins / trials,
      tieRate: ties / trials,
      lossRate: losses / trials,
      trials,
    };
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