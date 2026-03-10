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

type RankCounts = Map<number, number>;

export class HandEvaluator {
  private readonly rankMap: Record<string, number> = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    T: 10,
    J: 11,
    Q: 12,
    K: 13,
    A: 14,
  };

  public evaluate(cards: string[]): EvaluatedHand {
    if (cards.length < 5 || cards.length > 7) {
      throw new Error(`Expected 5 to 7 cards, got ${cards.length}`);
    }

    const combos = this.combinations(cards, 5);
    let best: EvaluatedHand | null = null;

    for (const combo of combos) {
      const evaluated = this.evaluateFiveCards(combo);
      if (!best || evaluated.rankValue > best.rankValue) {
        best = evaluated;
      }
    }

    if (!best) {
      throw new Error("Failed to evaluate hand.");
    }

    return best;
  }

  private evaluateFiveCards(cards: string[]): EvaluatedHand {
    const ranks = cards
      .map((c) => this.rankMap[c[0].toUpperCase()])
      .sort((a, b) => b - a);

    const suits = cards.map((c) => c[1].toLowerCase());
    const isFlush = suits.every((s) => s === suits[0]);

    const rankCounts = this.countRanks(ranks);
    const groups = [...rankCounts.entries()].sort((a, b) => {
      if (b[1] !== a[1]) return b[1] - a[1];
      return b[0] - a[0];
    });

    const straightHigh = this.getStraightHigh(ranks);
    const isStraight = straightHigh > 0;

    if (isFlush && isStraight) {
      return {
        category: "straight-flush",
        rankValue: this.encodeRank(8, [straightHigh]),
        bestFive: cards,
      };
    }

    if (groups[0][1] === 4) {
      const quad = groups[0][0];
      const kicker = groups[1][0];
      return {
        category: "four-of-a-kind",
        rankValue: this.encodeRank(7, [quad, kicker]),
        bestFive: cards,
      };
    }

    if (groups[0][1] === 3 && groups[1][1] === 2) {
      return {
        category: "full-house",
        rankValue: this.encodeRank(6, [groups[0][0], groups[1][0]]),
        bestFive: cards,
      };
    }

    if (isFlush) {
      return {
        category: "flush",
        rankValue: this.encodeRank(5, ranks),
        bestFive: cards,
      };
    }

    if (isStraight) {
      return {
        category: "straight",
        rankValue: this.encodeRank(4, [straightHigh]),
        bestFive: cards,
      };
    }

    if (groups[0][1] === 3) {
      const trips = groups[0][0];
      const kickers = groups
        .slice(1)
        .map(([r]) => r)
        .sort((a, b) => b - a);

      return {
        category: "three-of-a-kind",
        rankValue: this.encodeRank(3, [trips, ...kickers]),
        bestFive: cards,
      };
    }

    if (groups[0][1] === 2 && groups[1][1] === 2) {
      const highPair = Math.max(groups[0][0], groups[1][0]);
      const lowPair = Math.min(groups[0][0], groups[1][0]);
      const kicker = groups[2][0];

      return {
        category: "two-pair",
        rankValue: this.encodeRank(2, [highPair, lowPair, kicker]),
        bestFive: cards,
      };
    }

    if (groups[0][1] === 2) {
      const pair = groups[0][0];
      const kickers = groups
        .slice(1)
        .map(([r]) => r)
        .sort((a, b) => b - a);

      return {
        category: "pair",
        rankValue: this.encodeRank(1, [pair, ...kickers]),
        bestFive: cards,
      };
    }

    return {
      category: "high-card",
      rankValue: this.encodeRank(0, ranks),
      bestFive: cards,
    };
  }

  private countRanks(ranks: number[]): RankCounts {
    const counts = new Map<number, number>();
    for (const r of ranks) {
      counts.set(r, (counts.get(r) ?? 0) + 1);
    }
    return counts;
  }

  private getStraightHigh(ranksDesc: number[]): number {
    const unique = [...new Set(ranksDesc)].sort((a, b) => a - b);

    if (unique.includes(14)) {
      unique.unshift(1);
    }

    let run = 1;
    let bestHigh = 0;

    for (let i = 1; i < unique.length; i++) {
      if (unique[i] === unique[i - 1] + 1) {
        run++;
        if (run >= 5) {
          bestHigh = unique[i];
        }
      } else {
        run = 1;
      }
    }

    return bestHigh;
  }

  private encodeRank(category: number, values: number[]): number {
    let score = category * 1_000_000_000;
    let multiplier = 1_000_000;

    for (const v of values) {
      score += v * multiplier;
      multiplier = Math.floor(multiplier / 10);
      if (multiplier < 1) multiplier = 1;
    }

    return score;
  }

  private combinations<T>(arr: T[], k: number): T[][] {
    const result: T[][] = [];

    const dfs = (start: number, path: T[]) => {
      if (path.length === k) {
        result.push([...path]);
        return;
      }

      for (let i = start; i < arr.length; i++) {
        path.push(arr[i]);
        dfs(i + 1, path);
        path.pop();
      }
    };

    dfs(0, []);
    return result;
  }
}