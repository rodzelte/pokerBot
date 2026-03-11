/**
 * local-poker-engine.ts  (v2.2)
 *
 * Fixes vs v2.1:
 *  1. TIMEOUT FIX — draw + weakMade facing-bet paths now ALWAYS return.
 *     Previously they fell through to nothing, causing bot to stall/timeout.
 *  2. BLUFF DETECTION — tracks how many times villain has been the aggressor
 *     this hand. After 2+ bets we lower the call threshold to catch bluffs.
 *  3. RIVER BLUFF-CATCH — calls any small bet (≤40% pot) with any made hand
 *     on the river, since small river bets are frequently bluffs.
 *  4. RELAXED EQUITY THRESHOLDS — bot was folding too much because
 *     callEquityThreshold was compared against raw equity without pot-odds
 *     context. Now we always compare equity vs actual pot odds first.
 *  5. POT ALSO CONVERTED — table.getPot() returns raw chips; divides by
 *     bigBlind so SPR, pot-odds, and bet sizing are all in BBs.
 */
import {
  calculateEquityRealisation,
  type HandClass as EquityRealisationHandClass,
} from "./equity-realisation.ts";
import { BotAction, defaultCheckAction, defaultFoldAction } from "../interfaces/ai-client-interfaces.ts";
import { Game } from "../models/game.ts";
import { Table } from "../models/table.ts";
import { PlayerAction } from "../models/player-action.ts";
import { PlayerStats } from "../models/player-stats.ts";

import { HandEvaluator, HandCategory } from "./hand-evaluator.ts";
import { EquityEstimator, EquityResult } from "./equity-estimator.ts";

import {
  StrategyProfile,
  selectProfile,
  mixedStrategyFires,
} from "./gto-strategy-profiles.ts";

import {
  analyzeBoardTexture, BoardTexture,
  estimateRangeAdvantage,
  calcBlockerScore,
  potOddsEquity,
  minimumDefenceFrequency,
  bluffEV,
  estimateFoldEquity,
  getPreflopTier,
} from "./hand-range-utils.ts";

import { SessionTracker, HandOutcome } from "./session-tracker.ts";
import { OpponentModel, type OpponentRange } from "./opponent-range.ts";

// ─── Internal Types ───────────────────────────────────────────────────────────

type PreflopSpot =
  | "unopened" | "limped" | "raised"
  | "reraised" | "four_bet" | "five_bet_plus" | "jammed";

export type HandClass =
  | "air" | "draw" | "weakMade" | "strongMade" | "monster";

type Street = "preflop" | "flop" | "turn" | "river";

type VillainProfile = {
  totalHands: number;
  vpip: number;
  pfr: number;
};

type DecisionContext = {
  street:               Street;
  hand:                 string[];
  handKey:              string;
  board:                string[];
  heroId:               string;
  heroPosition:         string;
  heroStackBB:          number;
  effectiveStackBB:     number;
  potBB:                number;      // already in BBs
  actions:              PlayerAction[];
  villain:              VillainProfile | null;
  numPlayers:           number;
  spot?:                PreflopSpot;
  facingBet:            boolean;
  facingRaiseSizeBB:    number;      // already in BBs
  numRaisesThisStreet:  number;
  /** Aggression count — how many bet/raise actions villain has made this hand */
  villainAggressionCount: number;
  villainId:            string | null;
  texture:              BoardTexture;
  profile:              StrategyProfile;
  mixSeed:              string;
  bigBlind:             number;      // raw chip value of 1 BB
  /** Preflop equity vs 1 random opponent (from lookup table or MC). Always set. */
  preflopEquity:        number;
};

export type DecisionTrace = {
  handKey:      string;
  street:       string;
  position:     string;
  facingBet:    boolean;
  spot?:        string;
  handClass?:   string;
  chosenAction: string;
  chosenSizeBB: number;
  reasons:      string[];
  profile:      string;
  equity?:      number;
  potOdds?:     number;
  spr?:         number;
};

// ─── Constants ────────────────────────────────────────────────────────────────

const SPR_COMMITTED = 3;

const ABSOLUTE_PREMIUMS = new Set(["AA","KK","QQ","AKs","AKo"]);
const STRONG_CALL_HANDS = new Set(["JJ","TT","99","AQs","AQo","AJs","KQs"]);
const FOUR_BET_HANDS    = new Set(["AA","KK","QQ","JJ","AKs","AKo"]);
const JAM_CALL_HANDS    = new Set(["AA","KK","QQ","JJ","TT","AKs","AQs","AKo"]);

// ─── Engine ───────────────────────────────────────────────────────────────────

export class LocalPokerEngine {
  private readonly openRangeCache = new Map<string, Set<string>>();
  private readonly isoRangeCache  = new Map<string, Set<string>>();

  private readonly opponentModel    = new OpponentModel();
  private readonly villainRanges    = new Map<string, OpponentRange>();
  private readonly processedStreets = new Map<string, Set<string>>();

  private readonly handEvaluator   = new HandEvaluator();
  private readonly equityEstimator = new EquityEstimator();
  public  readonly sessionTracker  = new SessionTracker();

  private startingStackBB = 100;
  private lastTrace: DecisionTrace | null = null;



  
  // ── Public API ────────────────────────────────────────────────────────────

  public decide(game: Game, table: Table): BotAction {
    const ctx = this.buildDecisionContext(game, table);
    if (!ctx) return defaultFoldAction;
    return ctx.street === "preflop"
      ? this.decidePreflopCtx(ctx)
      : this.decidePostflopCtx(ctx);
  }

  public getLastTrace(): DecisionTrace | null { return this.lastTrace; }

  public setStartingStack(stackBB: number): void { this.startingStackBB = stackBB; }

  public recordHandOutcome(outcome: HandOutcome): void {
    this.sessionTracker.recordHand(outcome);
  }

  public printSessionSummary(): void { this.sessionTracker.printSummary(); }

  /**
   * Clear the opponent range and street-update log for a villain.
   * Call this at the start of each new hand so the model restarts fresh.
   */
  public resetVillainRange(villainId: string): void {
    this.villainRanges.delete(villainId);
    this.processedStreets.delete(villainId);
  }

  // ── Context Building ──────────────────────────────────────────────────────

  private buildDecisionContext(game: Game, table: Table): DecisionContext | null {
    const hero = game.getHero();
    if (!hero) return null;

    const hand        = hero.getHand() ?? [];
    const heroId      = hero.getPlayerId();
    const heroStackBB = hero.getStackSize();
    if (hand.length !== 2 || heroStackBB === undefined) return null;

    const bigBlind = game.getBigBlind(); // raw chips, e.g. 4

    const board    = this.getBoard(table);
    const actions  = table.getPlayerActions();
    const rawStreet = (table.getStreet() || "preflop").toLowerCase();
    const street: Street =
      rawStreet === "flop" || rawStreet === "turn" || rawStreet === "river"
        ? rawStreet : "preflop";

    const heroPosition     = table.getPlayerPositionFromId(heroId);
    const effectiveStackBB = this.getEffectiveStackBB(table, heroId, heroStackBB);
    const villain          = this.getMainVillainProfile(table, heroId);
    const villainId        = this.getMainVillainId(table, heroId);

    // Initialise range the first time we see this villain
    if (villainId && !this.villainRanges.has(villainId)) {
      this.villainRanges.set(villainId,
        this.opponentModel.createInitialRange(villain?.vpip ?? 40, villain?.pfr ?? 25));
    }

    const baseProfile = selectProfile(
      villain?.vpip ?? 0, villain?.pfr ?? 0, villain?.totalHands ?? 0, effectiveStackBB
    );
    const profile = this.sessionTracker.getAdaptiveProfile(
      baseProfile, heroStackBB, this.startingStackBB
    );

    const texture         = analyzeBoardTexture(board);
    // FIX: convert raw-chip raise and pot amounts → BBs
    const facingRaiseSizeBB = this.getLastRaiseSizeBB(actions, bigBlind);
    const potRawChips       = table.getPot();
    const potBB             = bigBlind > 0 ? potRawChips / bigBlind : potRawChips;
    const mixSeed           = `${this.normalizeHand(hand)}:${street}:${board.join(",")}`;

    // Count total bet/raise actions by villain this hand for bluff detection
    const villainAggressionCount = this.countVillainAggression(actions);

    // Preflop equity — always compute so every decision can use it
    const preflopEquity = this.equityEstimator.quickPreflopEquity(hand)
      ?? this.equityEstimator.estimateVsRandomRange(hand, [], 1).winRate;

    return {
      street,
      hand,
      handKey:              this.normalizeHand(hand),
      board,
      heroId,
      heroPosition,
      heroStackBB,
      effectiveStackBB,
      potBB,
      actions,
      villain,
      numPlayers:           table.getNumPlayers(),
      spot:                 street === "preflop"
                              ? this.classifyPreflopSpot(actions, bigBlind)
                              : undefined,
      facingBet:            this.isFacingLiveBet(actions),
      facingRaiseSizeBB,
      numRaisesThisStreet:  this.countRaisesThisStreet(actions),
      villainAggressionCount,
      villainId,
      texture,
      profile,
      mixSeed,
      bigBlind,
      preflopEquity,
    };
  }

  // ── Preflop ───────────────────────────────────────────────────────────────

  private decidePreflopCtx(ctx: DecisionContext): BotAction {
    const {
      handKey, spot, heroPosition, numPlayers, actions,
      effectiveStackBB, villain, facingRaiseSizeBB, potBB, profile, mixSeed,
    } = ctx;

    const openRange   = this.getOpenRange(heroPosition, numPlayers);
    const isoRange    = this.getIsoRange(heroPosition, numPlayers, profile);
    const defendRange = this.getDefendVsRaiseRange(heroPosition, effectiveStackBB, villain, profile);
    const jamRange    = this.getCallVsJamRange(effectiveStackBB, villain, profile);
    const limpers     = this.countLimpers(actions);
    const lastRaise   = facingRaiseSizeBB || 2.5;
    const tier        = getPreflopTier(handKey);
    const eq          = ctx.preflopEquity;

    // ── Unopened
    if (spot === "unopened") {
      if (openRange.has(handKey)) {
        // Dynamic sizing: premium equity (>72%) sizes up, speculative stays small
        const baseSize = this.getOpenSizeBB(heroPosition, numPlayers, effectiveStackBB);
        const openSize = eq >= 0.72 ? Math.min(baseSize * 1.3, effectiveStackBB)
                       : eq >= 0.60 ? baseSize
                       : Math.max(baseSize * 0.9, 2.0);
        return this.emit(ctx, "raise", openSize,
          ["open-range", tier, `eq:${eq.toFixed(2)}`]);
      }
      // Hands outside normal range but with decent equity can still open in late pos
      if (["BU","CO","SB"].includes(heroPosition) && eq >= 0.50) {
        const openSize = this.getOpenSizeBB(heroPosition, numPlayers, effectiveStackBB) * 0.9;
        return this.emit(ctx, "raise", Math.min(openSize, effectiveStackBB),
          ["open-equity-wide", `eq:${eq.toFixed(2)}`]);
      }
      if (heroPosition === "BB") return this.emit(ctx, "check", 0, ["bb-check-option"]);
      return this.emit(ctx, "fold", 0, ["outside-open-range", tier]);
    }

    // ── Limped
    if (spot === "limped") {
      if (isoRange.has(handKey)) {
        // Bigger iso-raise with strong equity to deny odds
        const baseIso = this.getIsoSizeBB(limpers, heroPosition, effectiveStackBB, numPlayers);
        const isoSize = eq >= 0.70 ? Math.min(baseIso * 1.25, effectiveStackBB) : baseIso;
        return this.emit(ctx, "raise", isoSize,
          ["iso-raise", `limpers:${limpers}`, tier, `eq:${eq.toFixed(2)}`]);
      }
      // Even outside iso-range, high equity hands should raise vs limpers
      if (eq >= 0.58 && limpers >= 1) {
        const isoSize = this.getIsoSizeBB(limpers, heroPosition, effectiveStackBB, numPlayers);
        return this.emit(ctx, "raise", isoSize,
          ["iso-raise-equity", `eq:${eq.toFixed(2)}`, `limpers:${limpers}`]);
      }
      if ((heroPosition === "BB" || heroPosition === "SB") &&
          (tier === "speculative" || tier === "medium")) {
        return this.emit(ctx, "check", 0, ["limp-along", tier]);
      }
      return this.emit(ctx, "fold", 0, ["iso-fold", tier]);
    }

    // ── Facing a single raise
    if (spot === "raised") {
      // Premiums + high-equity hands 3-bet for value
      if (ABSOLUTE_PREMIUMS.has(handKey) || eq >= 0.75) {
        const threeBetSize = eq >= 0.80
          ? Math.min(this.getThreeBetSizeBB(lastRaise, heroPosition, effectiveStackBB) * 1.15, effectiveStackBB)
          : this.getThreeBetSizeBB(lastRaise, heroPosition, effectiveStackBB);
        return this.emit(ctx, "raise", threeBetSize,
          ["3bet-value", tier, `eq:${eq.toFixed(2)}`]);
      }
      if (profile.mixed.lightThreeBet > 0 && tier !== "marginal" &&
          mixedStrategyFires(profile.mixed.lightThreeBet, mixSeed + ":light3b") &&
          defendRange.has(handKey) &&
          this.isLightThreeBetCandidate(handKey, heroPosition, villain)) {
        return this.emit(ctx, "raise",
          this.getThreeBetSizeBB(lastRaise, heroPosition, effectiveStackBB),
          ["3bet-light-mixed", tier]);
      }
      // Defend with adequate equity — be more willing with high equity
      if (defendRange.has(handKey) || eq >= 0.55) {
        const odds = potOddsEquity(potBB, lastRaise);
        if (eq >= odds * 0.80) {
          return this.emit(ctx, "call", 0, ["defend-call", `eq:${eq.toFixed(2)}`, tier]);
        }
      }
      return this.emit(ctx, "fold", 0, ["fold-to-raise", tier]);
    }

    // ── Facing a 3-bet
    if (spot === "reraised") {
      // Very high equity → always 4-bet for value
      if (FOUR_BET_HANDS.has(handKey) || eq >= 0.78) {
        if (effectiveStackBB <= 28)
          return this.emit(ctx, "raise", effectiveStackBB, ["4bet-jam", `eq:${eq.toFixed(2)}`]);
        return this.emit(ctx, "raise",
          this.getFourBetSizeBB(lastRaise, effectiveStackBB), ["4bet-value", tier, `eq:${eq.toFixed(2)}`]);
      }
      if (STRONG_CALL_HANDS.has(handKey) || eq >= 0.62) {
        const odds   = potOddsEquity(potBB, lastRaise);
        const spr    = this.calcSPR(effectiveStackBB, potBB + lastRaise);
        if (spr <= SPR_COMMITTED || eq >= odds * profile.impliedOddsFactor)
          return this.emit(ctx, "call", 0, ["3bet-call-strong", `eq:${eq.toFixed(2)}`]);
        return this.emit(ctx, "fold", 0, ["3bet-fold-strong-bad-odds", tier]);
      }
      if (effectiveStackBB <= 40 &&
          this.is4BetBluffCandidate(handKey, heroPosition) &&
          mixedStrategyFires(profile.mixed.lightThreeBet * 0.5, mixSeed + ":bluff4b")) {
        return this.emit(ctx, "raise",
          this.getFourBetSizeBB(lastRaise, effectiveStackBB), ["4bet-bluff-mixed", tier]);
      }
      return this.emit(ctx, "fold", 0, ["fold-to-3bet", tier]);
    }

    // ── Facing a 4-bet
    if (spot === "four_bet") {
      // Premium hands: 5-bet jam
      if (ABSOLUTE_PREMIUMS.has(handKey) ||
          (effectiveStackBB <= 20 && FOUR_BET_HANDS.has(handKey)))
        return this.emit(ctx, "raise", effectiveStackBB, ["5bet-jam", `eq:${eq.toFixed(2)}`]);
      // Strong hands can call if pot odds + SPR justify it
      if (STRONG_CALL_HANDS.has(handKey) || eq >= 0.68) {
        const odds   = potOddsEquity(potBB, lastRaise);
        const spr    = this.calcSPR(effectiveStackBB, potBB + lastRaise);
        if (spr <= SPR_COMMITTED || eq >= odds)
          return this.emit(ctx, "call", 0, ["4bet-call-strong", `eq:${eq.toFixed(2)}`, `spr:${spr.toFixed(1)}`]);
      }
      return this.emit(ctx, "fold", 0, ["fold-to-4bet", tier]);
    }

    // ── Facing a 5-bet or deeper
    if (spot === "five_bet_plus") {
      // Only the absolute best hands continue
      if (new Set(["AA","KK"]).has(handKey))
        return this.emit(ctx, "raise", effectiveStackBB, ["6bet-jam-nutted", `eq:${eq.toFixed(2)}`]);
      if (ABSOLUTE_PREMIUMS.has(handKey)) {
        const spr = this.calcSPR(effectiveStackBB, potBB + lastRaise);
        if (spr <= SPR_COMMITTED)
          return this.emit(ctx, "call", 0, ["5bet+-call-committed", `spr:${spr.toFixed(1)}`]);
        return this.emit(ctx, "call", 0, ["5bet+-call-premium", `eq:${eq.toFixed(2)}`]);
      }
      if (JAM_CALL_HANDS.has(handKey) || eq >= 0.75) {
        const odds = potOddsEquity(potBB, lastRaise);
        if (eq >= odds)
          return this.emit(ctx, "call", 0, ["5bet+-call-odds", `eq:${eq.toFixed(2)}`]);
      }
      return this.emit(ctx, "fold", 0, ["fold-to-5bet+", tier]);
    }

    // ── All-in
    if (spot === "jammed") {
      if (jamRange.has(handKey))
        return this.emit(ctx, "call", 0, ["jam-call", tier]);
      return this.emit(ctx, "fold", 0, ["fold-jam", tier]);
    }

    return defaultFoldAction;
  }

  // ── Postflop ──────────────────────────────────────────────────────────────

  private decidePostflopCtx(ctx: DecisionContext): BotAction {
    const {
      hand, board, numPlayers, effectiveStackBB, potBB,
      heroPosition, villain, facingBet, facingRaiseSizeBB,
      numRaisesThisStreet, street, texture, profile, mixSeed,
      villainAggressionCount, villainId,
    } = ctx;

    const handClass = this.classifyPostflopHand(hand, board, numPlayers);
    const madeHand  = this.evaluateMadeHand(hand, board);
    const madeCategory = madeHand?.category;
    const nutHand   = this.isNutHand(madeCategory);

    // ── Narrow opponent range once per street based on villain's action ──
    if (villainId && board.length >= 3) {
      const range = this.villainRanges.get(villainId);
      if (range) {
        if (!this.processedStreets.has(villainId)) {
          this.processedStreets.set(villainId, new Set());
        }
        const done = this.processedStreets.get(villainId)!;
        if (!done.has(street)) {
          done.add(street);
          this.opponentModel.updatePostflopRange(
            range, facingBet ? "bet" : "check", board,
            street as "flop" | "turn" | "river"
          );
        }
      }
    }

    const equity       = this.estimateEquity(hand, board, Math.max(1, numPlayers - 1));
    const villainRange = villainId ? this.villainRanges.get(villainId) : undefined;
    const rawEquity    = villainRange && board.length >= 3
      ? this.opponentModel.computeEquityVsRange(hand, villainRange, board)
      : equity?.winRate ?? 0;

const inPosition = ["BU","CO","HJ"].includes(heroPosition);
const isAggressor = ctx.villainAggressionCount === 0;

    const realisationHandClass = this.toEquityRealisationHandClass(
      handClass,
      rawEquity,
      hand,
      board
    );

    const eqVal = calculateEquityRealisation(
      rawEquity,
      realisationHandClass,
      inPosition,
      texture,
      isAggressor
    );

    const spr       = this.calcSPR(effectiveStackBB, potBB);
    const isRiver   = street === "river";
    const rangeAdv  = estimateRangeAdvantage(heroPosition, villain ? "BTN" : "CO", board, texture);
    const blockerScore = calcBlockerScore(hand, board);

    // ── Bluff detection adjustments ───────────────────────────────────────
    // Villain betting 2+ times this hand → lower our call threshold.
    // This stops them from running over us with pure air on every street.
    const isVillainBluffing = villainAggressionCount >= 2;
    // Small bets (≤40% pot) are statistically more likely to be bluffs
    const betFrac   = potBB > 0 ? facingRaiseSizeBB / potBB : 0.5;
    const isSmallBet = betFrac <= 0.40;
    // Bluff-adjusted call equity: lower threshold when villain is being aggressive
    const bluffAdjThreshold = isVillainBluffing
      ? Math.max(0.20, profile.callEquityThreshold - 0.10)
      : profile.callEquityThreshold;

    const baseReasons = [
      `hc:${madeHand?.category ?? "none"}`,
      `eq:${eqVal.toFixed(3)}`,
      `spr:${spr.toFixed(1)}`,
      `prof:${profile.name}`,
      ...(isVillainBluffing ? [`vagg:${villainAggressionCount}`] : []),
    ];

    // ══════════════════════════════════════════════════════════════════════
    // NOT FACING A BET — check or bet
    // ══════════════════════════════════════════════════════════════════════
    if (!facingBet) {
      if (handClass === "monster") {
        // Full-house+ → never slowplay, always bet big / overbet
        if (nutHand) {
          const nutBet = Math.min(effectiveStackBB, potBB * 1.5);
          return this.emit(ctx, "bet", nutBet,
            [...baseReasons, "nut-hand-overbet", `cat:${madeCategory}`], handClass);
        }
        if (mixedStrategyFires(profile.mixed.slowplayMonster, mixSeed + ":slow"))
          return this.emit(ctx, "check", 0, [...baseReasons, "slowplay-mixed"], handClass);
        return this.emit(ctx, "bet",
          this.getPostflopBetSize(potBB, effectiveStackBB, handClass, texture, profile, eqVal, spr),
          [...baseReasons, "monster-bet"], handClass);
      }

      if (handClass === "strongMade") {
        // High equity: bet aggressively, skip trapping
        if (eqVal >= 0.65) {
          return this.emit(ctx, "bet",
            this.getPostflopBetSize(potBB, effectiveStackBB, handClass, texture, profile, eqVal, spr),
            [...baseReasons, "value-bet-high-eq", `eq:${eqVal.toFixed(2)}`], handClass);
        }
        // Check-raise trap: only from OOP — don't waste IP betting opportunities
        if (!inPosition && mixedStrategyFires(profile.mixed.checkRaiseStrong, mixSeed + ":xr"))
          return this.emit(ctx, "check", 0, [...baseReasons, "xr-trap-mixed"], handClass);
        if (eqVal >= profile.valueBetMinEquity)
          return this.emit(ctx, "bet",
            this.getPostflopBetSize(potBB, effectiveStackBB, handClass, texture, profile, eqVal, spr),
            [...baseReasons, "value-bet"], handClass);
        // Even below valueBetMinEquity, bet when we have range advantage
        if (rangeAdv > 0.15)
          return this.emit(ctx, "bet",
            this.getPostflopBetSize(potBB, effectiveStackBB, handClass, texture, profile, eqVal, spr),
            [...baseReasons, "strong-range-pressure", `adv:${rangeAdv.toFixed(2)}`], handClass);
        // equity below threshold and no range advantage — check and re-evaluate
        return this.emit(ctx, "check", 0, [...baseReasons, "strong-check-low-eq"], handClass);
      }

      if (handClass === "draw" && !isRiver) {
        // Semi-bluff draws: fire when equity is reasonable
        if (eqVal >= profile.semiBluffMinEquity) {
          // Strong draws (flush/OESD) always bet; weaker draws use mixed strategy
          const isStrongDraw = eqVal >= 0.35 || (blockerScore >= 1 && inPosition);
          if (isStrongDraw || mixedStrategyFires(profile.mixed.gutshotSemiBluff, mixSeed + ":sbluff"))
            return this.emit(ctx, "bet",
              this.getPostflopBetSize(potBB, effectiveStackBB, handClass, texture, profile, eqVal, spr),
              [...baseReasons, "semi-bluff", `blk:${blockerScore}`], handClass);
        }
        // Even below semiBluffMinEquity, bet draws with range advantage for balance
        if (rangeAdv > 0.15 && mixedStrategyFires(profile.mixed.flopCBetAir, mixSeed + ":draw-probe"))
          return this.emit(ctx, "bet",
            this.getPostflopBetSize(potBB, effectiveStackBB, "air", texture, profile, eqVal, spr),
            [...baseReasons, "draw-probe-bet", `adv:${rangeAdv.toFixed(2)}`], handClass);
        return this.emit(ctx, "check", 0, [...baseReasons, "draw-check"], handClass);
      }

      // Pair/weakMade: dynamically switch between passive and aggressive
      if (handClass === "weakMade") {
        // High equity weakMade (e.g. TPTK): bet for value even on dry boards
        if (eqVal >= 0.50) {
          return this.emit(ctx, "bet",
            this.getPostflopBetSize(potBB, effectiveStackBB, "weakMade", texture, profile, eqVal, spr),
            [...baseReasons, "weak-value-bet", `eq:${eqVal.toFixed(2)}`], handClass);
        }
        // Range-pressure: bet weak hands when we have clear range advantage
        if (rangeAdv > 0.20 && !isRiver) {
          return this.emit(ctx, "bet",
            this.getPostflopBetSize(potBB, effectiveStackBB, "weakMade", texture, profile, eqVal, spr),
            [...baseReasons, "range-pressure-bet", `adv:${rangeAdv.toFixed(2)}`], handClass);
        }
        // Protection bet on wet boards or with some equity
        if (!isRiver && eqVal >= 0.38)
          return this.emit(ctx, "bet",
            this.getPostflopBetSize(potBB, effectiveStackBB, "weakMade", texture, profile, eqVal, spr),
            [...baseReasons, "protection-bet"], handClass);
        if (street === "flop" && texture.wetScore < 2 && eqVal < 0.38)
          return this.emit(ctx, "check", 0, [...baseReasons, "pair-check-flop"], handClass);
        return this.emit(ctx, "check", 0, [...baseReasons, "weak-check"], handClass);
      }

      // Air c-bet on boards where we have range advantage or as positional pressure
      if (handClass === "air" && !isRiver) {
        const cbFreq = street === "flop" ? profile.mixed.flopCBetAir
                     : street === "turn" ? profile.mixed.turnCBetAir
                     : profile.mixed.riverBluff;
        // Fire c-bet: range advantage lowers the bar, but even without it we
        // can range-bet at frequency when in position or on favourable textures
        const canCBet = rangeAdv > 0.05
          || (rangeAdv > -0.05 && (blockerScore >= 1 || inPosition))
          || texture.hasHighCard;
        if (canCBet && mixedStrategyFires(cbFreq, mixSeed + ":cbet"))
          return this.emit(ctx, "bet",
            this.getPostflopBetSize(potBB, effectiveStackBB, "air", texture, profile, eqVal, spr),
            [...baseReasons, "range-cbet", `adv:${rangeAdv.toFixed(2)}`], handClass);
      }

      // River bluff with blockers
      if (handClass === "air" && isRiver && blockerScore >= 1) {
        const foldEq  = estimateFoldEquity(villain?.vpip ?? 40, villain?.pfr ?? 20,
                          villain?.totalHands ?? 0, 0.75, texture);
        const betSize = potBB * 0.75;
        if (bluffEV(potBB, betSize, foldEq) > 0 &&
            mixedStrategyFires(profile.mixed.riverBluff, mixSeed + ":rbluff"))
          return this.emit(ctx, "bet", betSize,
            [...baseReasons, `river-bluff`, `foldEq:${foldEq.toFixed(2)}`], handClass);
      }

      return this.emit(ctx, "check", 0, [...baseReasons, "check-default"], handClass);
    }

    // ══════════════════════════════════════════════════════════════════════
    // FACING A BET / RAISE
    // Every branch below MUST return — no fallthrough allowed.
    // ══════════════════════════════════════════════════════════════════════
    const potOdds = potOddsEquity(potBB, facingRaiseSizeBB);
    const mdf     = minimumDefenceFrequency(potBB, facingRaiseSizeBB);

    // ── Raise-war detection ───────────────────────────────────────────────
    // When the street has seen 3+ raises, we're in an escalating raise war.
    // Only the absolute best hands should continue — everything else folds.
    const inRaiseWar = numRaisesThisStreet >= 3;

    if (inRaiseWar) {
      // In a raise war (3+ raises on one street), only nut hands continue
      if (handClass === "monster" && nutHand)
        return this.emit(ctx, "raise", effectiveStackBB,
          [...baseReasons, "raise-war-nut-jam", `raises:${numRaisesThisStreet}`], handClass);
      if (handClass === "monster")
        return this.emit(ctx, "call", 0,
          [...baseReasons, "raise-war-monster-call", `raises:${numRaisesThisStreet}`], handClass);
      if (handClass === "strongMade" && spr <= SPR_COMMITTED)
        return this.emit(ctx, "call", 0,
          [...baseReasons, "raise-war-committed", `raises:${numRaisesThisStreet}`], handClass);
      if (handClass === "strongMade" && eqVal >= 0.65)
        return this.emit(ctx, "call", 0,
          [...baseReasons, "raise-war-strong-high-eq", `eq:${eqVal.toFixed(2)}`], handClass);
      // Everything else: fold — raise wars represent monster-vs-monster ranges
      return this.emit(ctx, "fold", 0,
        [...baseReasons, "raise-war-fold", `raises:${numRaisesThisStreet}`], handClass);
    }

    // ── Monster: raise or call ────────────────────────────────────────────
    if (handClass === "monster") {
      // Full-house+ → always raise/jam, never flat
      if (nutHand) {
        const jamSize = effectiveStackBB;
        return this.emit(ctx, "raise", jamSize,
          [...baseReasons, "nut-hand-jam", `cat:${madeCategory}`], handClass);
      }
      if (numRaisesThisStreet < 2)
        return this.emit(ctx, "raise",
          this.getPostflopValueRaiseBB(potBB, facingRaiseSizeBB, effectiveStackBB, spr),
          [...baseReasons, "monster-raise"], handClass);
      // 2 raises already: flat-call to trap or jam if short
      if (spr <= SPR_COMMITTED)
        return this.emit(ctx, "raise", effectiveStackBB,
          [...baseReasons, "monster-jam-low-spr"], handClass);
      return this.emit(ctx, "call", 0, [...baseReasons, "monster-call-capped"], handClass);
    }

    // ── Strong made hand ──────────────────────────────────────────────────
    if (handClass === "strongMade") {
      // Facing a re-raise (2 raises): tighten considerably — need strong equity
      if (numRaisesThisStreet >= 2) {
        if (spr <= SPR_COMMITTED)
          return this.emit(ctx, "call", 0, [...baseReasons, "strong-committed-reraise"], handClass);
        if (eqVal >= 0.55)
          return this.emit(ctx, "call", 0, [...baseReasons, "strong-call-reraise", `eq:${eqVal.toFixed(2)}`], handClass);
        return this.emit(ctx, "fold", 0, [...baseReasons, "strong-fold-reraise", `eq:${eqVal.toFixed(2)}`], handClass);
      }
      if (spr <= SPR_COMMITTED)
        return this.emit(ctx, "call", 0, [...baseReasons, "strong-committed"], handClass);
      // Equity clears pot odds (with implied-odds discount) → call or x/r
      if (eqVal >= potOdds * profile.impliedOddsFactor) {
        if (numRaisesThisStreet === 0 &&
            mixedStrategyFires(profile.mixed.checkRaiseStrong * 0.5, mixSeed + ":xr2"))
          return this.emit(ctx, "raise",
            this.getPostflopValueRaiseBB(potBB, facingRaiseSizeBB, effectiveStackBB, spr),
            [...baseReasons, "xr-value"], handClass);
        return this.emit(ctx, "call", 0, [...baseReasons, "strong-call", `eq:${eqVal.toFixed(2)}`], handClass);
      }
      // Bluff-catch: villain has been aggressive + small bet → call even with marginal equity
      if (isVillainBluffing && isSmallBet && eqVal >= 0.30)
        return this.emit(ctx, "call", 0, [...baseReasons, "anti-bluff-strong"], handClass);
      return this.emit(ctx, "fold", 0, [...baseReasons, "strong-fold-bad-odds"], handClass);
    }

    // ── Draw ──────────────────────────────────────────────────────────────
    if (handClass === "draw") {
      // Facing a re-raise with a draw: almost always fold (not getting odds)
      if (numRaisesThisStreet >= 2) {
        if (spr <= SPR_COMMITTED)
          return this.emit(ctx, "call", 0, [...baseReasons, "draw-committed-reraise"], handClass);
        return this.emit(ctx, "fold", 0, [...baseReasons, "draw-fold-reraise", `raises:${numRaisesThisStreet}`], handClass);
      }
      if (!isRiver) {
        if (spr <= SPR_COMMITTED)
          return this.emit(ctx, "call", 0, [...baseReasons, "draw-committed"], handClass);
        if (eqVal >= potOdds * profile.impliedOddsFactor)
          return this.emit(ctx, "call", 0, [...baseReasons, "draw-call", `impl:${(potOdds * profile.impliedOddsFactor).toFixed(2)}`], handClass);
        // Semi-bluff raise on draws vs passive villains with blockers
        if (eqVal >= 0.30 && blockerScore >= 1 && numRaisesThisStreet < 2 &&
            villain && villain.vpip > 40 && villain.pfr < 20)
          return this.emit(ctx, "raise",
            this.getPostflopValueRaiseBB(potBB, facingRaiseSizeBB, effectiveStackBB, spr),
            [...baseReasons, "draw-semibluff-raise"], handClass);
        // Bluff-catch with draw vs small bet on aggressive villain
        if (isVillainBluffing && isSmallBet)
          return this.emit(ctx, "call", 0, [...baseReasons, "anti-bluff-draw-call"], handClass);
        return this.emit(ctx, "fold", 0, [...baseReasons, "draw-fold-bad-price"], handClass);
      } else {
        // River: draw missed — bluff-catch only if villain bet small and we have
        // any showdown value (e.g. ace-high or a pair picked up along the way)
        if (isSmallBet && isVillainBluffing && eqVal >= 0.25)
          return this.emit(ctx, "call", 0, [...baseReasons, "river-bluffcatch-busted-draw"], handClass);
        return this.emit(ctx, "fold", 0, [...baseReasons, "draw-river-missed"], handClass);
      }
    }

    // ── Weak made hand ────────────────────────────────────────────────────
    if (handClass === "weakMade") {
      // Facing a re-raise with a weak hand: fold unless pot-committed
      if (numRaisesThisStreet >= 2) {
        if (spr <= SPR_COMMITTED)
          return this.emit(ctx, "call", 0, [...baseReasons, "weak-committed-reraise"], handClass);
        return this.emit(ctx, "fold", 0, [...baseReasons, "weak-fold-reraise", `raises:${numRaisesThisStreet}`], handClass);
      }
      if (spr <= SPR_COMMITTED)
        return this.emit(ctx, "call", 0, [...baseReasons, "committed-spr"], handClass);
      // Pot odds are met — call
      if (eqVal >= potOdds)
        return this.emit(ctx, "call", 0, [...baseReasons, "weak-equity-call", `eq:${eqVal.toFixed(2)}`], handClass);
      // River: small bet with any pair — bluff-catch (villain often bets small as a bluff)
      if (isRiver && isSmallBet && eqVal >= 0.35)
        return this.emit(ctx, "call", 0, [...baseReasons, "river-bluffcatch-weak"], handClass);
      // Multi-street aggressor + small bet → call
      if (isVillainBluffing && isSmallBet && eqVal >= 0.30)
        return this.emit(ctx, "call", 0, [...baseReasons, "anti-bluff-weak"], handClass);
      return this.emit(ctx, "fold", 0, [...baseReasons, "weak-fold", `eq:${eqVal.toFixed(2)}`], handClass);
    }

    // ── Air / no made hand ────────────────────────────────────────────────
    // (handClass === "air") — explicit block, always returns
    {
      // Facing a re-raise with air: always fold (no bluff-catching air vs reraises)
      if (numRaisesThisStreet >= 2)
        return this.emit(ctx, "fold", 0, [...baseReasons, "air-fold-reraise", `raises:${numRaisesThisStreet}`], handClass);
      if (spr <= SPR_COMMITTED)
        return this.emit(ctx, "call", 0, [...baseReasons, "committed-spr"], handClass);
      // Equity clears bluff-adjusted threshold
      if (eqVal >= bluffAdjThreshold)
        return this.emit(ctx, "call", 0, [...baseReasons, "mdf-defend", `mdf:${mdf.toFixed(2)}`], handClass);
      // Very aggressive villain betting small — hero can call as pure bluff-catch
      if (isVillainBluffing && isSmallBet && villainAggressionCount >= 3 && eqVal >= 0.20)
        return this.emit(ctx, "call", 0, [...baseReasons, "air-bluffcatch-extreme-aggressor"], handClass);
      return this.emit(ctx, "fold", 0, [...baseReasons, "fold-insufficient"], handClass);
    }
  }

  // ── Spot Classification ───────────────────────────────────────────────────

  private classifyPreflopSpot(actions: PlayerAction[], bigBlind: number): PreflopSpot {
    const voluntary = actions.filter((a) => a.getAction().toLowerCase() !== "post");
    const raises = voluntary.filter((a) => {
      const act = a.getAction().toLowerCase();
      return act === "raise" || act === "bet";
    });
    const calls = voluntary.filter((a) => a.getAction().toLowerCase() === "call");

    if (raises.length === 0 && calls.length === 0) return "unopened";
    if (raises.length === 0 && calls.length > 0)   return "limped";

    const biggestChips = Math.max(...raises.map((a) => a.getBetAmount()), 0);
    const biggestBB    = bigBlind > 0 ? biggestChips / bigBlind : biggestChips;

    if (biggestBB >= 15) return "jammed";   // all-in or near-all-in shove
    if (raises.length >= 4) return "five_bet_plus";
    if (raises.length >= 3) return "four_bet";
    if (raises.length >= 2) return "reraised";
    return "raised";
  }

  // ── Villain Aggression Counter ────────────────────────────────────────────

  /**
   * Count how many bet/raise actions have occurred this hand.
   * Used as a proxy for villain bluffing — if they've bet 2+ times,
   * we suspect multi-street bluffing and loosen our call requirements.
   */
  private countVillainAggression(actions: PlayerAction[]): number {
    return actions.filter((a) => {
      const act = a.getAction().toLowerCase();
      return act === "bet" || act === "raise";
    }).length;
  }

  // ── Sizing ────────────────────────────────────────────────────────────────

  private getOpenSizeBB(pos: string, numPlayers: number, effBB: number): number {
    if (effBB <= 12) return effBB;
    if (numPlayers === 2 && pos === "SB") return Math.min(2.5, effBB);
    const sizes: Record<string, number> = {
      SB:3.0, BU:2.3, CO:2.3, HJ:2.5, LJ:2.5, MP:2.5, "UTG+1":2.5, UTG:2.5,
    };
    return Math.min(sizes[pos] ?? 2.5, effBB);
  }

  private getIsoSizeBB(limpers: number, pos: string, effBB: number, numPlayers: number): number {
    if (effBB <= 12) return effBB;
    let size = 3.5 + Math.max(0, limpers - 1);
    if (pos === "BB" || pos === "SB") size += 0.5;
    if (numPlayers === 2 && pos === "BB") size = 4.0;
    return Math.min(size, effBB);
  }

  private getThreeBetSizeBB(lastRaiseBB: number, pos: string, effBB: number): number {
    if (effBB <= 20) return effBB;
    const mult = ["BU","CO","HJ"].includes(pos) ? 3.0 : 4.0;
    return Math.min(Math.max(lastRaiseBB * mult, 7.5), effBB);
  }

  private getFourBetSizeBB(threeBetBB: number, effBB: number): number {
    if (effBB <= 25) return effBB;
    return Math.min(Math.max(threeBetBB * 2.5, 18), effBB);
  }

  private getPostflopBetSize(
    potBB: number, effBB: number, handClass: HandClass,
    texture: BoardTexture, profile: StrategyProfile, equity: number, spr: number
  ): number {
    if (effBB <= 10) return effBB;
    if (spr <= 2.5)  return Math.min(effBB, potBB);
    let fraction: number;
    if (handClass === "monster")
      fraction = texture.wetScore >= 3 ? profile.sizing.overbetFraction
               : texture.wetScore >= 2 ? profile.sizing.largeFraction
               : profile.sizing.medFraction;
    else if (handClass === "strongMade")
      fraction = texture.wetScore >= 2 ? profile.sizing.largeFraction : profile.sizing.medFraction;
    else if (handClass === "weakMade")
      fraction = profile.sizing.smallFraction;
    else
      fraction = texture.wetScore >= 2 ? profile.sizing.medFraction : profile.sizing.smallFraction;

    // Dynamic equity scaling: bet bigger when equity is high, smaller when marginal
    if (equity >= 0.80) fraction = Math.min(fraction * 1.35, profile.sizing.overbetFraction);
    else if (equity >= 0.65) fraction = Math.min(fraction * 1.15, profile.sizing.largeFraction);
    else if (equity < 0.40) fraction = Math.max(fraction * 0.8, profile.sizing.smallFraction);

    const kellyAdj = this.sessionTracker.kellyBetAdjustment(
      equity, potBB, potBB * fraction, profile.kellyFraction
    );
    return Math.min(Math.max(potBB * fraction * kellyAdj, 1), effBB);
  }

  private getPostflopValueRaiseBB(
    potBB: number, betBB: number, effBB: number, spr: number
  ): number {
    if (effBB <= 10 || spr <= 2) return effBB;
    return Math.min(Math.max(betBB * 2.8, potBB * 0.9), effBB);
  }

  // ── Ranges ────────────────────────────────────────────────────────────────

  private getOpenRange(position: string, numPlayers: number): Set<string> {
    const key = `${position}:${numPlayers}`;
    if (this.openRangeCache.has(key)) return this.openRangeCache.get(key)!;

    const ranges: Record<string, string[]> = {
      "UTG":   ["AA","KK","QQ","JJ","TT","99","AKs","AQs","AJs","KQs","AKo","AQo"],
      "UTG+1": ["AA","KK","QQ","JJ","TT","99","88","AKs","AQs","AJs","KQs","AKo","AQo","AJo"],
      "MP":    ["AA","KK","QQ","JJ","TT","99","88","77","AKs","AQs","AJs","ATs","KQs","QJs","JTs","AKo","AQo","AJo","KQo"],
      "LJ":    ["AA","KK","QQ","JJ","TT","99","88","77","66","AKs","AQs","AJs","ATs","KQs","KJs","QJs","JTs","T9s","AKo","AQo","AJo","KQo"],
      "HJ":    ["AA","KK","QQ","JJ","TT","99","88","77","66","55","AKs","AQs","AJs","ATs","A9s","KQs","KJs","QJs","JTs","T9s","98s","AKo","AQo","AJo","KQo","QJo"],
      "CO":    ["AA","KK","QQ","JJ","TT","99","88","77","66","55","44","AKs","AQs","AJs","ATs","A9s","A8s","KQs","KJs","KTs","QJs","QTs","JTs","T9s","98s","87s","AKo","AQo","AJo","ATo","KQo","KJo","QJo"],
      "BU":    ["AA","KK","QQ","JJ","TT","99","88","77","66","55","44","33","22","AKs","AQs","AJs","ATs","A9s","A8s","A7s","A6s","A5s","A4s","A3s","A2s","KQs","KJs","KTs","K9s","QJs","QTs","Q9s","JTs","J9s","T9s","98s","87s","76s","AKo","AQo","AJo","ATo","KQo","KJo","QJo","JTo"],
      "SB":    ["AA","KK","QQ","JJ","TT","99","88","77","66","55","44","33","22","AKs","AQs","AJs","ATs","A9s","A8s","A7s","A6s","A5s","A4s","A3s","A2s","KQs","KJs","KTs","K9s","QJs","QTs","Q9s","JTs","J9s","T9s","98s","87s","76s","65s","AKo","AQo","AJo","ATo","KQo","KJo","QJo","JTo"],
      "BB":    [],
    };

    const range = (numPlayers === 2 && position === "SB")
      ? new Set(["AA","KK","QQ","JJ","TT","99","88","77","66","55",
                 "AKs","AQs","AJs","ATs","A9s","A8s","A7s","A6s","A5s","A4s","A3s","A2s",
                 "KQs","KJs","KTs","QJs","QTs","JTs","T9s","98s","87s","76s",
                 "AKo","AQo","AJo","ATo","KQo","KJo","QJo","JTo"])
      : new Set(ranges[position] ?? []);

    this.openRangeCache.set(key, range);
    return range;
  }

  private getIsoRange(position: string, numPlayers: number, profile: StrategyProfile): Set<string> {
    const key = `${position}:${numPlayers}:${profile.name}`;
    if (this.isoRangeCache.has(key)) return this.isoRangeCache.get(key)!;
    const base  = this.getOpenRange(position, numPlayers);
    const extra = (profile.name === "loose" || profile.name === "lag")
      ? ["KTo","QTo","97s","86s","75s","54s","A7o","A6o","K9o","Q9o","J9o","T8s"]
      : ["KTo","QTo","97s","86s","75s","54s"];
    const range = new Set([...base, ...extra]);
    this.isoRangeCache.set(key, range);
    return range;
  }

  private getDefendVsRaiseRange(
    position: string, effBB: number,
    villain: VillainProfile | null, profile: StrategyProfile
  ): Set<string> {
    let range: Set<string>;
    if (position === "BB") {
      range = new Set(["AA","KK","QQ","JJ","TT","99","88","77","66",
        "AKs","AQs","AJs","ATs","A9s","KQs","KJs","QJs","JTs","T9s","98s",
        "AKo","AQo","AJo","KQo","QJo"]);
    } else if (["SB","BU","CO"].includes(position)) {
      range = new Set(["AA","KK","QQ","JJ","TT","99","88",
        "AKs","AQs","AJs","ATs","KQs","QJs","JTs","AKo","AQo","AJo","KQo"]);
    } else {
      range = new Set(["AA","KK","QQ","JJ","TT","99","AKs","AQs","AJs","KQs","AKo","AQo"]);
    }
    if (effBB <= 20) {
      range = new Set([...range].filter((h) =>
        ["AA","KK","QQ","JJ","TT","99","AKs","AQs","AKo","AQo"].includes(h)));
    }
    if (villain && villain.totalHands >= 20 && villain.vpip > 45 && villain.pfr > 25)
      ["A9s","KTs","QTs","J9s","T8s","ATo","KJo"].forEach((h) => range.add(h));
    if (profile.name === "loose" || profile.name === "lag")
      ["A8s","A7s","K9s","Q9s","J9s","87s"].forEach((h) => range.add(h));
    return range;
  }

  private getCallVsJamRange(
    effBB: number, villain: VillainProfile | null, profile: StrategyProfile
  ): Set<string> {
    const range = new Set<string>(["AA","KK","QQ","JJ","TT","AKs","AQs","AKo"]);
    if (effBB <= 15) ["99","88","AQo","AJs"].forEach((h) => range.add(h));
    if (effBB <= 10) ["77","ATs","KQs","AJo"].forEach((h) => range.add(h));
    if (villain && villain.totalHands >= 20 && villain.vpip > 50 && villain.pfr > 35)
      ["99","88","AQo","AJs","ATs"].forEach((h) => range.add(h));
    if (profile.name === "loose" || profile.name === "lag") {
      range.add("99"); range.add("AQo");
    }
    return range;
  }

  // ── Candidate Selection ───────────────────────────────────────────────────

  private isLightThreeBetCandidate(
    handKey: string, pos: string, villain: VillainProfile | null
  ): boolean {
    if (!["BU","CO","HJ"].includes(pos)) return false;
    const lightHands = new Set([
      "AJs","A5s","A4s","KQs","KJs","QJs","JTs","T9s","98s","87s","76s","65s",
    ]);
    if (!lightHands.has(handKey)) return false;
    if (villain && villain.totalHands >= 15 && villain.pfr > 22) return true;
    return true;
  }

  private is4BetBluffCandidate(handKey: string, pos: string): boolean {
    return new Set(["AJs","A5s","A4s","A3s","KQs","KJs"]).has(handKey) &&
      ["BU","CO"].includes(pos);
  }

  // ── Hand Classification ───────────────────────────────────────────────────

  private classifyPostflopHand(hand: string[], board: string[], numPlayers: number): HandClass {
    const madeHand = this.evaluateMadeHand(hand, board);
    const equity   = this.estimateEquity(hand, board, Math.max(1, numPlayers - 1));
    if (madeHand) {
      const baseClass = this.mapMadeHandToClass(madeHand.category, equity, hand, board);
      // Dynamic upgrade: if equity says we're very strong, play more aggressively
      if (baseClass === "weakMade" && equity && equity.winRate >= 0.65) return "strongMade";
      if (baseClass === "strongMade" && equity && equity.winRate >= 0.85) return "monster";
      return baseClass;
    }
    // Strong draws with high equity play like made hands
    const hasFlush = this.hasFlushDraw(hand, board);
    const hasOESD  = this.hasOpenEndedStraightDraw(hand, board);
    if ((hasFlush && hasOESD) || (equity && equity.winRate >= 0.50 && board.length < 5))
      return "strongMade"; // combo draw / massive draw plays like a made hand
    if (hasFlush || hasOESD)                        return "draw";
    if (this.hasGutshotStraightDraw(hand, board))   return "draw";
    return "air";
  }

  private evaluateMadeHand(hand: string[], board: string[]) {
    if (board.length < 3) return null;
    return this.handEvaluator.evaluate([...hand, ...board]);
  }

  private estimateEquity(
    hand: string[], board: string[], numOpponents: number
  ): EquityResult | null {
    if (board.length < 3) return null;
    const trials = board.length >= 5 ? 400 : board.length >= 4 ? 600 : 800;
    return this.equityEstimator.estimateVsRandomRange(
      hand, board, Math.max(1, numOpponents), trials
    );
  }

  private mapMadeHandToClass(
    category: HandCategory, equity: EquityResult | null,
    hand: string[], board: string[]
  ): HandClass {
    if (["straight-flush","four-of-a-kind","full-house"].includes(category)) return "monster";
    if (["flush","straight","three-of-a-kind"].includes(category)) return "strongMade";
    if (category === "two-pair") return "strongMade";
    if (category === "pair") {
      if (this.isTopPairOrOverpair(hand, board)) return "weakMade";
      if (equity && equity.winRate >= 0.55) return "weakMade";
      return "weakMade";
    }
    if (equity && equity.winRate >= 0.28 && board.length < 5) return "draw";
    return "air";
  }

  /**
   * Returns true for full-house+, signalling the engine to use
   * maximum-aggression bet sizing (overbets, no slowplay).
   */
  private isNutHand(category: HandCategory | undefined): boolean {
    if (!category) return false;
    return ["full-house", "four-of-a-kind", "straight-flush"].includes(category);
  }

  private toEquityRealisationHandClass(
  handClass: HandClass,
  rawEquity: number,
  hand: string[],
  board: string[]
): EquityRealisationHandClass {

  if (handClass !== "draw") {
    return handClass as EquityRealisationHandClass;
  }

  const hasFlushDraw  = this.hasFlushDraw(hand, board);
  const hasOESD       = this.hasOpenEndedStraightDraw(hand, board);
  const hasGutshot    = this.hasGutshotStraightDraw(hand, board);

  const isStrongDraw =
    hasFlushDraw ||
    hasOESD ||
    rawEquity >= 0.30;

  if (isStrongDraw) return "strongDraw";

  if (hasGutshot) return "weakDraw";

  return "weakDraw";
}


  // ── Utilities ─────────────────────────────────────────────────────────────

  private calcSPR(effBB: number, potBB: number): number {
    return potBB > 0 ? effBB / potBB : 999;
  }

  private countRaisesThisStreet(actions: PlayerAction[]): number {
    return actions.filter((a) => {
      const act = a.getAction().toLowerCase();
      return act === "raise" || act === "bet";
    }).length;
  }

  private isFacingLiveBet(actions: PlayerAction[]): boolean {
    for (let i = actions.length - 1; i >= 0; i--) {
      const act = actions[i].getAction().toLowerCase();
      if (act === "bet" || act === "raise") return true;
      if (act === "check") return false;
    }
    return false;
  }

  private countLimpers(actions: PlayerAction[]): number {
    if (actions.some((a) => ["raise","bet"].includes(a.getAction().toLowerCase()))) return 0;
    return actions.filter((a) => a.getAction().toLowerCase() === "call").length;
  }

  private getLastRaiseSizeBB(actions: PlayerAction[], bigBlind: number): number {
    const raises = actions.filter((a) => ["raise","bet"].includes(a.getAction().toLowerCase()));
    if (!raises.length) return 0;
    const raw = raises[raises.length - 1].getBetAmount();
    return bigBlind > 0 ? raw / bigBlind : raw;
  }

  private getBoard(table: Table): string[] {
    const runout = table.getRunout();
    if (!runout) return [];
    return runout.replace(/[\[\]]/g, "").split(",").map((s) => s.trim()).filter(Boolean);
  }

  private getEffectiveStackBB(table: Table, heroId: string, heroStackBB: number): number {
    const others = Array.from(table.getPlayerInitialStacks().entries())
      .filter(([id]) => id !== heroId)
      .map(([, s]) => s as number);
    return others.length ? Math.min(heroStackBB, ...others) : heroStackBB;
  }

  private getMainVillainProfile(table: Table, heroId: string): VillainProfile | null {
    for (const [id, name] of table.getIdToName().entries()) {
      if (id === heroId) continue;
      try { return this.toVillainProfile(table.getPlayerStatsFromName(name)); }
      catch { continue; }
    }
    return null;
  }

  private getMainVillainId(table: Table, heroId: string): string | null {
    for (const [id] of table.getIdToName().entries()) {
      if (id !== heroId) return id;
    }
    return null;
  }

  private toVillainProfile(stats: PlayerStats): VillainProfile {
    const total = stats.getTotalHands();
    return {
      totalHands: total,
      vpip: total > 0 ? (stats.getVPIPHands() / total) * 100 : 0,
      pfr:  total > 0 ? (stats.getPFRHands()  / total) * 100 : 0,
    };
  }

  private normalizeHand(cards: string[]): string {
    const order = "AKQJT98765432";
    const [a, b] = cards;
    const [hi, lo] = [a[0].toUpperCase(), b[0].toUpperCase()]
      .sort((x, y) => order.indexOf(x) - order.indexOf(y));
    if (hi === lo) return hi + lo;
    return hi + lo + (a[1] === b[1] ? "s" : "o");
  }

  private getBoardTopRank(board: string[]): string {
    const order = "AKQJT98765432";
    return [...board.map((c) => c[0].toUpperCase())]
      .sort((a, b) => order.indexOf(a) - order.indexOf(b))[0] ?? "";
  }

  private isTopPairOrOverpair(hand: string[], board: string[]): boolean {
    if (!board.length) return false;
    const top    = this.getBoardTopRank(board);
    const holes  = hand.map((c) => c[0].toUpperCase());
    if (holes.includes(top)) return true;
    if (holes[0] === holes[1]) {
      return "AKQJT98765432".indexOf(holes[0]) < "AKQJT98765432".indexOf(top);
    }
    return false;
  }

  private hasFlushDraw(hand: string[], board: string[]): boolean {
    const counts = new Map<string, number>();
    for (const c of [...hand, ...board])
      counts.set(c[1], (counts.get(c[1]) ?? 0) + 1);
    return [...counts.values()].some((v) => v === 4);
  }

  private getUniqueSortedRanks(cards: string[]): number[] {
    const map: Record<string,number> = {
      A:14,K:13,Q:12,J:11,T:10,"9":9,"8":8,"7":7,"6":6,"5":5,"4":4,"3":3,"2":2,
    };
    const vals = new Set<number>();
    for (const c of cards) {
      const v = map[c[0].toUpperCase()];
      if (v) { vals.add(v); if (c[0].toUpperCase() === "A") vals.add(1); }
    }
    return [...vals].sort((a, b) => a - b);
  }

  private hasOpenEndedStraightDraw(hand: string[], board: string[]): boolean {
    const vals = this.getUniqueSortedRanks([...hand, ...board]);
    for (let i = 0; i <= vals.length - 4; i++)
      if (vals[i + 3] - vals[i] === 3) return true;
    return false;
  }

  private hasGutshotStraightDraw(hand: string[], board: string[]): boolean {
    const vals = this.getUniqueSortedRanks([...hand, ...board]);
    for (let i = 0; i <= vals.length - 4; i++)
      if (vals[i + 3] - vals[i] === 4) return true;
    return false;
  }

  // ── Emit ──────────────────────────────────────────────────────────────────

  private emit(
    ctx: DecisionContext,
    action: string,
    sizeBB: number,
    reasons: string[],
    handClass?: HandClass
  ): BotAction {
    const equity = ctx.board.length >= 3
      ? this.estimateEquity(ctx.hand, ctx.board, Math.max(1, ctx.numPlayers - 1))?.winRate
      : ctx.preflopEquity;

    this.lastTrace = {
      handKey:      ctx.handKey,
      street:       ctx.street,
      position:     ctx.heroPosition,
      facingBet:    ctx.facingBet,
      spot:         ctx.spot,
      handClass,
      chosenAction: action,
      chosenSizeBB: sizeBB,
      reasons,
      profile:      ctx.profile.name,
      equity,
      potOdds:      ctx.facingBet && ctx.facingRaiseSizeBB > 0
                      ? potOddsEquity(ctx.potBB, ctx.facingRaiseSizeBB)
                      : undefined,
      spr:          ctx.potBB > 0 ? ctx.effectiveStackBB / ctx.potBB : undefined,
    };

    if (action === "fold")  return defaultFoldAction;
    if (action === "check") return defaultCheckAction;
    return { action_str: action as any, bet_size_in_BBs: sizeBB };
  }
}