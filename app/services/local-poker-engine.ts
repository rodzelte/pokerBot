import { BotAction, defaultCheckAction, defaultFoldAction } from "../interfaces/ai-client-interfaces.ts";
import { Game } from "../models/game.ts";
import { Table } from "../models/table.ts";
import { PlayerAction } from "../models/player-action.ts";
import { PlayerStats } from "../models/player-stats.ts";

import { HandEvaluator, HandCategory } from "./hand-evaluator.ts";
import { EquityEstimator, EquityResult } from "./equity-estimator.ts";

type PreflopSpot =
  | "unopened"
  | "limped"
  | "raised"
  | "reraised"
  | "jammed";

type HandClass =
  | "air"
  | "draw"
  | "weakMade"
  | "strongMade"
  | "monster";

type Street = "preflop" | "flop" | "turn" | "river";

type Position =
  | "SB"
  | "BB"
  | "UTG"
  | "UTG+1"
  | "MP"
  | "LJ"
  | "HJ"
  | "CO"
  | "BU";

type VillainProfile = {
  totalHands: number;
  vpip: number;
  pfr: number;
};

type DecisionContext = {
  street: Street;
  hand: string[];
  handKey: string;
  board: string[];
  heroId: string;
  heroPosition: string;
  heroStackBB: number;
  effectiveStackBB: number;
  potBB: number;
  actions: PlayerAction[];
  villain: VillainProfile | null;
  numPlayers: number;
  spot?: PreflopSpot;
  facingBet: boolean;
};

type DecisionTrace = {
  handKey: string;
  street: string;
  position: string;
  facingBet: boolean;
  spot?: string;
  handClass?: string;
  chosenAction: string;
  chosenSizeBB: number;
  reasons: string[];
};

export class LocalPokerEngine {
  private readonly openRangeCache = new Map<string, Set<string>>();
  private readonly isoRangeCache = new Map<string, Set<string>>();

    private readonly handEvaluator = new HandEvaluator();
  private readonly equityEstimator = new EquityEstimator();

  private lastTrace: DecisionTrace | null = null;

  public decide(game: Game, table: Table): BotAction {
    const ctx = this.buildDecisionContext(game, table);
    if (!ctx) return defaultFoldAction;

    if (ctx.street === "preflop") {
      return this.decidePreflopCtx(ctx);
    }

    return this.decidePostflopCtx(ctx);
  }

  public getLastTrace(): DecisionTrace | null {
    return this.lastTrace;
  }

  private setTrace(trace: DecisionTrace): void {
    this.lastTrace = trace;
  }
  private evaluateMadeHand(hand: string[], board: string[]) {
    if (board.length < 3) {
      return null;
    }

    return this.handEvaluator.evaluate([...hand, ...board]);
  }

  private estimateEquity(hand: string[], board: string[], numOpponents: number): EquityResult | null {
    if (board.length < 3) {
      return null;
    }

    const safeOpponents = Math.max(1, numOpponents);
    const trials = board.length >= 4 ? 1200 : 800;

    return this.equityEstimator.estimateVsRandomRange(
      hand,
      board,
      safeOpponents,
      trials
    );
  }

  private getApproxOpponentCount(numPlayers: number): number {
    return Math.max(1, numPlayers - 1);
  }

  private mapMadeHandToClass(
    category: HandCategory,
    equity: EquityResult | null,
    hand: string[],
    board: string[]
  ): HandClass {
    if (
      category === "straight-flush" ||
      category === "four-of-a-kind" ||
      category === "full-house"
    ) {
      return "monster";
    }

    if (
      category === "flush" ||
      category === "straight" ||
      category === "three-of-a-kind" ||
      category === "two-pair"
    ) {
      return "strongMade";
    }

    if (category === "pair") {
      if (this.isTopPairOrOverpair(hand, board)) {
        return "strongMade";
      }

      if (equity && equity.winRate >= 0.55) {
        return "strongMade";
      }

      return "weakMade";
    }

    if (equity && equity.winRate >= 0.30 && board.length < 5) {
      return "draw";
    }

    return "air";
  }
  private buildDecisionContext(game: Game, table: Table): DecisionContext | null {
    const hero = game.getHero();
    if (!hero) return null;

    const hand = hero.getHand() ?? [];
    const heroId = hero.getPlayerId();
    const heroStackBB = hero.getStackSize();

    if (hand.length !== 2 || heroStackBB === undefined) {
      return null;
    }


    

    const board = this.getBoard(table);
    const actions = table.getPlayerActions();
    const rawStreet = (table.getStreet() || "preflop").toLowerCase();
    const street: Street =
      rawStreet === "flop" || rawStreet === "turn" || rawStreet === "river"
        ? rawStreet
        : "preflop";

    const heroPosition = table.getPlayerPositionFromId(heroId);
    const effectiveStackBB = this.getEffectiveStackBB(table, heroId, heroStackBB);
    const villain = this.getMainVillainProfile(table, heroId);

    
    return {
      street,
      hand,
      handKey: this.normalizeHand(hand),
      board,
      heroId,
      heroPosition,
      heroStackBB,
      effectiveStackBB,
      potBB: table.getPot(),
      actions,
      villain,
      numPlayers: table.getNumPlayers(),
      spot: street === "preflop" ? this.classifyPreflopSpot(actions) : undefined,
      facingBet: this.isFacingLiveBet(actions),
    };
  }

  private decidePreflopCtx(ctx: DecisionContext): BotAction {
    const { handKey, spot, heroPosition, numPlayers, actions, effectiveStackBB, villain } = ctx;

    const openRange = this.getOpenRange(heroPosition, numPlayers);
    const isoRange = this.getIsoRange(heroPosition, numPlayers);
    const defendRange = this.getDefendVsRaiseRange(heroPosition, effectiveStackBB, villain);
    const jamCallRange = this.getCallVsJamRange(effectiveStackBB, villain);

    const limpers = this.countLimpers(actions);
    const lastRaiseSizeBB = this.getLastRaiseSizeBB(actions);

    if (spot === "unopened") {
      if (openRange.has(handKey)) {
        const size = this.getOpenSizeBB(heroPosition, numPlayers, effectiveStackBB);
        this.setTrace({
          handKey,
          street: "preflop",
          position: heroPosition,
          facingBet: false,
          spot,
          chosenAction: "raise",
          chosenSizeBB: size,
          reasons: ["open-range", `stack:${effectiveStackBB.toFixed(1)}bb`],
        });
        return {
          action_str: "raise",
          bet_size_in_BBs: size,
        };
      }

      if (heroPosition === "BB") {
        this.setTrace({
          handKey,
          street: "preflop",
          position: heroPosition,
          facingBet: false,
          spot,
          chosenAction: "check",
          chosenSizeBB: 0,
          reasons: ["bb-check-option"],
        });
        return defaultCheckAction;
      }

      this.setTrace({
        handKey,
        street: "preflop",
        position: heroPosition,
        facingBet: false,
        spot,
        chosenAction: "fold",
        chosenSizeBB: 0,
        reasons: ["outside-open-range"],
      });
      return defaultFoldAction;
    }

    if (spot === "limped") {
      if (isoRange.has(handKey)) {
        const size = this.getIsoSizeBB(limpers, heroPosition, effectiveStackBB, numPlayers);
        this.setTrace({
          handKey,
          street: "preflop",
          position: heroPosition,
          facingBet: false,
          spot,
          chosenAction: "raise",
          chosenSizeBB: size,
          reasons: ["iso-range", `limpers:${limpers}`],
        });
        return {
          action_str: "raise",
          bet_size_in_BBs: size,
        };
      }

      const passiveAction = heroPosition === "BB" ? "check" : "check";
      this.setTrace({
        handKey,
        street: "preflop",
        position: heroPosition,
        facingBet: false,
        spot,
        chosenAction: passiveAction,
        chosenSizeBB: 0,
        reasons: ["no-iso"],
      });
      return defaultCheckAction;
    }

    if (spot === "raised" || spot === "reraised") {
      if (!defendRange.has(handKey)) {
        this.setTrace({
          handKey,
          street: "preflop",
          position: heroPosition,
          facingBet: true,
          spot,
          chosenAction: "fold",
          chosenSizeBB: 0,
          reasons: ["outside-defend-range"],
        });
        return defaultFoldAction;
      }

      if (this.shouldThreeBetForTraining(handKey, effectiveStackBB, villain, heroPosition)) {
        const size = this.getThreeBetSizeBB(lastRaiseSizeBB || 2.5, heroPosition, effectiveStackBB);
        this.setTrace({
          handKey,
          street: "preflop",
          position: heroPosition,
          facingBet: true,
          spot,
          chosenAction: "raise",
          chosenSizeBB: size,
          reasons: ["value-reraise-training"],
        });
        return {
          action_str: "raise",
          bet_size_in_BBs: size,
        };
      }

      this.setTrace({
        handKey,
        street: "preflop",
        position: heroPosition,
        facingBet: true,
        spot,
        chosenAction: "call",
        chosenSizeBB: 0,
        reasons: ["defend-range"],
      });
      return {
        action_str: "call",
        bet_size_in_BBs: 0,
      };
    }

    if (spot === "jammed") {
      if (jamCallRange.has(handKey)) {
        this.setTrace({
          handKey,
          street: "preflop",
          position: heroPosition,
          facingBet: true,
          spot,
          chosenAction: "call",
          chosenSizeBB: 0,
          reasons: ["jam-call-range"],
        });
        return {
          action_str: "call",
          bet_size_in_BBs: 0,
        };
      }

      this.setTrace({
        handKey,
        street: "preflop",
        position: heroPosition,
        facingBet: true,
        spot,
        chosenAction: "fold",
        chosenSizeBB: 0,
        reasons: ["outside-jam-call-range"],
      });
      return defaultFoldAction;
    }

    return defaultFoldAction;
  }

  private decidePostflopCtx(ctx: DecisionContext): BotAction {
    const handClass = this.classifyPostflopHand(ctx.hand, ctx.board, ctx.numPlayers);

        const madeHand = this.evaluateMadeHand(ctx.hand, ctx.board);
    const equity = this.estimateEquity(
      ctx.hand,
      ctx.board,
      this.getApproxOpponentCount(ctx.numPlayers)
    );

    if (!ctx.facingBet) {
      if (handClass === "monster" || handClass === "strongMade" || handClass === "draw") {
        const size = this.getPostflopBetSizeBB(
          ctx.potBB,
          ctx.effectiveStackBB,
          handClass,
          ctx.board
        );

        this.setTrace({
          handKey: ctx.handKey,
          street: ctx.street,
          position: ctx.heroPosition,
          facingBet: false,
          handClass,
          chosenAction: "bet",
          chosenSizeBB: size,
          reasons: [
  "initiative-training",
  `made:${madeHand?.category ?? "none"}`,
  `eq:${equity ? equity.winRate.toFixed(3) : "n/a"}`
],
        });

        return {
          action_str: "bet",
          bet_size_in_BBs: size,
        };
      }

      this.setTrace({
        handKey: ctx.handKey,
        street: ctx.street,
        position: ctx.heroPosition,
        facingBet: false,
        handClass,
        chosenAction: "check",
        chosenSizeBB: 0,
        reasons: [
  "medium-or-air",
  `made:${madeHand?.category ?? "none"}`,
  `eq:${equity ? equity.winRate.toFixed(3) : "n/a"}`
],
      });
      return defaultCheckAction;
    }

    if (handClass === "monster" || handClass === "strongMade") {
      this.setTrace({
        handKey: ctx.handKey,
        street: ctx.street,
        position: ctx.heroPosition,
        facingBet: true,
        handClass,
        chosenAction: "call",
        chosenSizeBB: 0,
       reasons: [
  "made-hand-continue",
  `made:${madeHand?.category ?? "none"}`,
  `eq:${equity ? equity.winRate.toFixed(3) : "n/a"}`
],
      });
      return {
        action_str: "call",
        bet_size_in_BBs: 0,
      };
    }

    if (
      handClass === "draw" &&
      ((ctx.villain && ctx.villain.totalHands >= 20 && ctx.villain.vpip > 40) ||
        ctx.effectiveStackBB <= 20)
    ) {
      this.setTrace({
        handKey: ctx.handKey,
        street: ctx.street,
        position: ctx.heroPosition,
        facingBet: true,
        handClass,
        chosenAction: "call",
        chosenSizeBB: 0,
        reasons: [
  "draw-continue-training",
  `made:${madeHand?.category ?? "none"}`,
  `eq:${equity ? equity.winRate.toFixed(3) : "n/a"}`
],
      });
      return {
        action_str: "call",
        bet_size_in_BBs: 0,
      };
    }

    if (ctx.heroPosition === "BB" && handClass === "weakMade") {
      this.setTrace({
        handKey: ctx.handKey,
        street: ctx.street,
        position: ctx.heroPosition,
        facingBet: true,
        handClass,
        chosenAction: "call",
        chosenSizeBB: 0,
        reasons: [
  "bb-defend-light",
  `made:${madeHand?.category ?? "none"}`,
  `eq:${equity ? equity.winRate.toFixed(3) : "n/a"}`
],
      });
      return {
        action_str: "call",
        bet_size_in_BBs: 0,
      };
    }

    this.setTrace({
      handKey: ctx.handKey,
      street: ctx.street,
      position: ctx.heroPosition,
      facingBet: true,
      handClass,
      chosenAction: "fold",
      chosenSizeBB: 0,
      reasons: [
  "insufficient-strength",
  `made:${madeHand?.category ?? "none"}`,
  `eq:${equity ? equity.winRate.toFixed(3) : "n/a"}`
],
    });
    return defaultFoldAction;
  }

  private classifyPreflopSpot(actions: PlayerAction[]): PreflopSpot {
    const voluntary = actions.filter((a) => a.getAction().toLowerCase() !== "post");

    const raises = voluntary.filter((a) => {
      const action = a.getAction().toLowerCase();
      return action === "raise" || action === "bet";
    });

    const calls = voluntary.filter((a) => a.getAction().toLowerCase() === "call");

    if (raises.length === 0 && calls.length === 0) {
      return "unopened";
    }

    if (raises.length === 0 && calls.length > 0) {
      return "limped";
    }

    const biggestRaise = Math.max(...raises.map((a) => a.getBetAmount()), 0);

    if (biggestRaise >= 10) {
      return "jammed";
    }

    if (raises.length >= 2) {
      return "reraised";
    }

    return "raised";
  }

  private isFacingLiveBet(actions: PlayerAction[]): boolean {
    if (!actions.length) return false;

    for (let i = actions.length - 1; i >= 0; i--) {
      const action = actions[i].getAction().toLowerCase();

      if (action === "bet" || action === "raise") {
        return true;
      }

      if (action === "check") {
        return false;
      }
    }

    return false;
  }

  private getOpenRange(position: string, numPlayers: number): Set<string> {
    const key = `${position}:${numPlayers}`;
    const cached = this.openRangeCache.get(key);
    if (cached) return cached;

    let range: Set<string>;

    if (numPlayers === 2 && position === "SB") {
      range = new Set([
        "AA","KK","QQ","JJ","TT","99","88","77","66","55",
        "AKs","AQs","AJs","ATs","A9s","A8s","A7s","A6s","A5s","A4s","A3s","A2s",
        "KQs","KJs","KTs","QJs","QTs","JTs","T9s","98s","87s","76s",
        "AKo","AQo","AJo","ATo","KQo","KJo","QJo","JTo"
      ]);
    } else {
      const ranges: Record<string, string[]> = {
        "UTG": ["AA","KK","QQ","JJ","TT","99","AKs","AQs","AJs","KQs","AKo","AQo"],
        "UTG+1": ["AA","KK","QQ","JJ","TT","99","88","AKs","AQs","AJs","KQs","AKo","AQo","AJo"],
        "MP": ["AA","KK","QQ","JJ","TT","99","88","77","AKs","AQs","AJs","ATs","KQs","QJs","JTs","AKo","AQo","AJo","KQo"],
        "LJ": ["AA","KK","QQ","JJ","TT","99","88","77","66","AKs","AQs","AJs","ATs","KQs","KJs","QJs","JTs","T9s","AKo","AQo","AJo","KQo"],
        "HJ": ["AA","KK","QQ","JJ","TT","99","88","77","66","55","AKs","AQs","AJs","ATs","A9s","KQs","KJs","QJs","JTs","T9s","98s","AKo","AQo","AJo","KQo","QJo"],
        "CO": ["AA","KK","QQ","JJ","TT","99","88","77","66","55","44","AKs","AQs","AJs","ATs","A9s","A8s","KQs","KJs","KTs","QJs","QTs","JTs","T9s","98s","87s","AKo","AQo","AJo","ATo","KQo","KJo","QJo"],
        "BU": ["AA","KK","QQ","JJ","TT","99","88","77","66","55","44","33","22","AKs","AQs","AJs","ATs","A9s","A8s","A7s","A6s","A5s","A4s","A3s","A2s","KQs","KJs","KTs","K9s","QJs","QTs","Q9s","JTs","J9s","T9s","98s","87s","76s","AKo","AQo","AJo","ATo","KQo","KJo","QJo","JTo"],
        "SB": ["AA","KK","QQ","JJ","TT","99","88","77","66","55","44","33","22","AKs","AQs","AJs","ATs","A9s","A8s","A7s","A6s","A5s","A4s","A3s","A2s","KQs","KJs","KTs","K9s","QJs","QTs","Q9s","JTs","J9s","T9s","98s","87s","76s","65s","AKo","AQo","AJo","ATo","KQo","KJo","QJo","JTo"],
        "BB": [],
      };

      range = new Set(ranges[position] ?? []);
    }

    this.openRangeCache.set(key, range);
    return range;
  }

  private getIsoRange(position: string, numPlayers: number): Set<string> {
    const key = `${position}:${numPlayers}`;
    const cached = this.isoRangeCache.get(key);
    if (cached) return cached;

    const base = this.getOpenRange(position, numPlayers);
    const extra = ["KTo", "QTo", "97s", "86s", "75s", "54s"];
    if (numPlayers === 2) {
      extra.push("A7o", "A6o", "K9o", "Q9o", "J9o");
    }

    const range = new Set([...base, ...extra]);
    this.isoRangeCache.set(key, range);
    return range;
  }

  private getDefendVsRaiseRange(
    position: string,
    effectiveStackBB: number,
    villain: VillainProfile | null
  ): Set<string> {
    let range = new Set<string>();

    if (position === "BB") {
      range = new Set([
        "AA","KK","QQ","JJ","TT","99","88","77","66",
        "AKs","AQs","AJs","ATs","A9s","KQs","KJs","QJs","JTs","T9s","98s",
        "AKo","AQo","AJo","KQo","QJo"
      ]);
    } else if (position === "SB" || position === "BU" || position === "CO") {
      range = new Set([
        "AA","KK","QQ","JJ","TT","99","88",
        "AKs","AQs","AJs","ATs","KQs","QJs","JTs",
        "AKo","AQo","AJo","KQo"
      ]);
    } else {
      range = new Set([
        "AA","KK","QQ","JJ","TT","99",
        "AKs","AQs","AJs","KQs",
        "AKo","AQo"
      ]);
    }

    if (effectiveStackBB <= 20) {
      range = new Set(
        [...range].filter((h) =>
          ["AA","KK","QQ","JJ","TT","99","AKs","AQs","AKo","AQo"].includes(h)
        )
      );
    }

    if (villain && villain.totalHands >= 20 && villain.vpip > 45 && villain.pfr > 25) {
      ["A9s","KTs","QTs","J9s","T8s","ATo","KJo"].forEach((h) => range.add(h));
    }

    return range;
  }

  private getCallVsJamRange(effectiveStackBB: number, villain: VillainProfile | null): Set<string> {
    const range = new Set<string>(["AA","KK","QQ","JJ","TT","AKs","AQs","AKo"]);

    if (effectiveStackBB <= 15) {
      ["99","88","AQo","AJs"].forEach((h) => range.add(h));
    }

    if (villain && villain.totalHands >= 20 && villain.vpip > 50 && villain.pfr > 35) {
      ["99","88","AQo","AJs","ATs"].forEach((h) => range.add(h));
    }

    return range;
  }

  private shouldThreeBetForTraining(
    handKey: string,
    effectiveStackBB: number,
    villain: VillainProfile | null,
    heroPosition: string
  ): boolean {
    if (["AA","KK","QQ","AKs","AKo"].includes(handKey)) {
      return true;
    }

    if (effectiveStackBB <= 25 && ["JJ","TT","AQs","AQo"].includes(handKey)) {
      return true;
    }

    const inPosition = ["BU", "CO", "HJ"].includes(heroPosition);
    if (
      inPosition &&
      villain &&
      villain.totalHands >= 20 &&
      villain.vpip > 45 &&
      villain.pfr > 30
    ) {
      if (["JJ","TT","AQs","AJs","KQs"].includes(handKey)) {
        return true;
      }
    }

    return false;
  }

  private getIsoSizeBB(
    limpers: number,
    heroPosition: string,
    effectiveStackBB: number,
    numPlayers: number
  ): number {
    if (effectiveStackBB <= 12) {
      return effectiveStackBB;
    }

    let size = 3.5 + Math.max(0, limpers - 1);

    if (heroPosition === "BB" || heroPosition === "SB") {
      size += 0.5;
    }

    if (numPlayers === 2 && heroPosition === "BB") {
      size = 4.0;
    }

    return Math.min(size, effectiveStackBB);
  }

  private getOpenSizeBB(position: string, numPlayers: number, effectiveStackBB: number): number {
    if (effectiveStackBB <= 12) {
      return effectiveStackBB;
    }

    if (numPlayers === 2 && position === "SB") {
      return Math.min(2.5, effectiveStackBB);
    }

    switch (position) {
      case "SB":
        return Math.min(3.0, effectiveStackBB);
      case "BU":
      case "CO":
        return Math.min(2.3, effectiveStackBB);
      case "HJ":
      case "LJ":
      case "MP":
      case "UTG":
      case "UTG+1":
        return Math.min(2.5, effectiveStackBB);
      default:
        return Math.min(2.5, effectiveStackBB);
    }
  }

  private getThreeBetSizeBB(
    lastRaiseSizeBB: number,
    heroPosition: string,
    effectiveStackBB: number
  ): number {
    if (effectiveStackBB <= 20) {
      return effectiveStackBB;
    }

    const inPosition =
      heroPosition === "BU" ||
      heroPosition === "CO" ||
      heroPosition === "HJ";

    const multiplier = inPosition ? 3.0 : 4.0;
    const size = lastRaiseSizeBB * multiplier;

    return Math.min(Math.max(size, 7.5), effectiveStackBB);
  }

  private isWetBoard(board: string[]): boolean {
    if (board.length < 3) return false;

    const suits = new Map<string, number>();
    const ranks = board.map((c) => c[0].toUpperCase());
    const order = "A23456789TJQKA";
    const idxs = ranks
      .map((r) => order.indexOf(r))
      .filter((i) => i >= 0)
      .sort((a, b) => a - b);

    for (const card of board) {
      const suit = card[1];
      suits.set(suit, (suits.get(suit) ?? 0) + 1);
    }

    const twoTone = Array.from(suits.values()).some((v) => v >= 2);

    let connected = false;
    for (let i = 0; i < idxs.length - 1; i++) {
      if (idxs[i + 1] - idxs[i] <= 2) {
        connected = true;
        break;
      }
    }

    return twoTone || connected;
  }

  private getPostflopBetSizeBB(
    potBB: number,
    effectiveStackBB: number,
    handClass: HandClass,
    board: string[]
  ): number {
    if (effectiveStackBB <= 10) {
      return effectiveStackBB;
    }

    const wetBoard = this.isWetBoard(board);
    let fraction = 0.33;

    if (handClass === "monster") {
      fraction = wetBoard ? 0.75 : 0.66;
    } else if (handClass === "strongMade") {
      fraction = wetBoard ? 0.66 : 0.5;
    } else if (handClass === "draw") {
      fraction = wetBoard ? 0.5 : 0.4;
    }

    const size = potBB * fraction;
    return Math.min(Math.max(size, 1), effectiveStackBB);
  }

  private classifyPostflopHand(
    hand: string[],
    board: string[],
    numPlayers: number = 2
  ): HandClass {
    const madeHand = this.evaluateMadeHand(hand, board);
    const equity = this.estimateEquity(
      hand,
      board,
      this.getApproxOpponentCount(numPlayers)
    );

    if (madeHand) {
      return this.mapMadeHandToClass(madeHand.category, equity, hand, board);
    }

    const hasFlushDraw = this.hasFlushDraw(hand, board);
    const hasOESD = this.hasOpenEndedStraightDraw(hand, board);
    const hasGutshot = this.hasGutshotStraightDraw(hand, board);

    if (hasFlushDraw || hasOESD || hasGutshot) {
      return "draw";
    }

    return "air";
  }

  private getBoard(table: Table): string[] {
    const runout = table.getRunout();
    if (!runout) return [];
    return runout
      .replace(/[\[\]]/g, "")
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
  }

  private getEffectiveStackBB(table: Table, heroId: string, heroStackBB: number): number {
    const villainStacks = Array.from(table.getPlayerInitialStacks().entries())
      .filter(([id]) => id !== heroId)
      .map(([, stack]) => stack as number);

    if (!villainStacks.length) return heroStackBB;
    return Math.min(heroStackBB, ...villainStacks);
  }

  private getMainVillainProfile(table: Table, heroId: string): VillainProfile | null {
    for (const [id, name] of table.getIdToName().entries()) {
      if (id === heroId) continue;
      try {
        const stats = table.getPlayerStatsFromName(name);
        return this.toVillainProfile(stats);
      } catch {
        continue;
      }
    }
    return null;
  }

  private countLimpers(actions: PlayerAction[]): number {
    const voluntaryCalls = actions.filter((a) => a.getAction().toLowerCase() === "call");
    const raises = actions.filter((a) => {
      const action = a.getAction().toLowerCase();
      return action === "raise" || action === "bet";
    });

    if (raises.length > 0) {
      return 0;
    }

    return voluntaryCalls.length;
  }

  private getLastRaiseSizeBB(actions: PlayerAction[]): number {
    const raises = actions.filter((a) => {
      const action = a.getAction().toLowerCase();
      return action === "raise" || action === "bet";
    });

    if (!raises.length) {
      return 0;
    }

    return raises[raises.length - 1].getBetAmount();
  }

  private toVillainProfile(stats: PlayerStats): VillainProfile {
    const totalHands = stats.getTotalHands();
    const vpip = totalHands > 0 ? (stats.getVPIPHands() / totalHands) * 100 : 0;
    const pfr = totalHands > 0 ? (stats.getPFRHands() / totalHands) * 100 : 0;

    return {
      totalHands,
      vpip,
      pfr,
    };
  }

  private normalizeHand(cards: string[]): string {
    const [a, b] = cards;
    const r1 = a[0].toUpperCase();
    const r2 = b[0].toUpperCase();
    const s1 = a[1];
    const s2 = b[1];

    const order = "AKQJT98765432";
    const sorted = [r1, r2].sort((x, y) => order.indexOf(x) - order.indexOf(y));
    const hi = sorted[0];
    const lo = sorted[1];

    if (hi === lo) return hi + lo;
    return hi + lo + (s1 === s2 ? "s" : "o");
  }

  private getRankCounts(cards: string[]): Map<string, number> {
    const counts = new Map<string, number>();
    for (const card of cards) {
      const rank = card[0].toUpperCase();
      counts.set(rank, (counts.get(rank) ?? 0) + 1);
    }
    return counts;
  }

  private getBoardTopRank(board: string[]): string {
    const order = "AKQJT98765432";
    const boardRanks = board.map((c) => c[0].toUpperCase());
    boardRanks.sort((a, b) => order.indexOf(a) - order.indexOf(b));
    return boardRanks[0] ?? "";
  }

  private hasFlush(cards: string[]): boolean {
    const suits = new Map<string, number>();
    for (const card of cards) {
      const suit = card[1];
      suits.set(suit, (suits.get(suit) ?? 0) + 1);
    }
    return Array.from(suits.values()).some((v) => v >= 5);
  }

  private hasFlushDraw(hand: string[], board: string[]): boolean {
    const suits = new Map<string, number>();
    for (const card of [...hand, ...board]) {
      const suit = card[1];
      suits.set(suit, (suits.get(suit) ?? 0) + 1);
    }
    return Array.from(suits.values()).some((v) => v === 4);
  }

  private getUniqueSortedRanks(cards: string[]): number[] {
    const map: Record<string, number> = {
      A: 14,
      K: 13,
      Q: 12,
      J: 11,
      T: 10,
      "9": 9,
      "8": 8,
      "7": 7,
      "6": 6,
      "5": 5,
      "4": 4,
      "3": 3,
      "2": 2,
    };

    const vals = new Set<number>();
    for (const c of cards) {
      const r = c[0].toUpperCase();
      const v = map[r];
      if (v) vals.add(v);
      if (r === "A") vals.add(1);
    }

    return Array.from(vals).sort((a, b) => a - b);
  }

  private hasStraight(cards: string[]): boolean {
    const values = this.getUniqueSortedRanks(cards);
    for (let i = 0; i <= values.length - 5; i++) {
      if (values[i + 4] - values[i] === 4) {
        return true;
      }
    }
    return false;
  }

  private hasOpenEndedStraightDraw(hand: string[], board: string[]): boolean {
    const values = this.getUniqueSortedRanks([...hand, ...board]);
    for (let i = 0; i <= values.length - 4; i++) {
      const window = values.slice(i, i + 4);
      if (window[3] - window[0] === 3) {
        return true;
      }
    }
    return false;
  }

  private hasGutshotStraightDraw(hand: string[], board: string[]): boolean {
    const values = this.getUniqueSortedRanks([...hand, ...board]);
    for (let i = 0; i <= values.length - 4; i++) {
      const window = values.slice(i, i + 4);
      if (window[3] - window[0] === 4) {
        return true;
      }
    }
    return false;
  }

  private isTopPairOrOverpair(hand: string[], board: string[]): boolean {
    if (board.length === 0) return false;

    const boardTop = this.getBoardTopRank(board);
    const holeRanks = hand.map((c) => c[0].toUpperCase());

    if (holeRanks.includes(boardTop)) {
      return true;
    }

    if (holeRanks[0] === holeRanks[1]) {
      const pairRank = holeRanks[0];
      const order = "AKQJT98765432";
      return order.indexOf(pairRank) < order.indexOf(boardTop);
    }

    return false;
  }
}