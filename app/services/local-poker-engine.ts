/**
 * local-poker-engine.ts  (v2.0)
 *
 * A GTO-approximating, exploitative, self-adjusting poker decision engine.
 *
 * Architecture:
 *   ┌─────────────────────────────────────────┐
 *   │  decide()                               │
 *   │    buildDecisionContext()               │
 *   │    selectProfile()  ← villain stats     │
 *   │    sessionTracker.getAdaptiveProfile()  │  ← session learning
 *   │    decidePreflopCtx() / decidePostflopCtx()
 *   │      ├─ range lookups                  │
 *   │      ├─ equity (MC + lookup table)     │
 *   │      ├─ pot-odds / MDF / Kelly sizing  │
 *   │      ├─ mixed-strategy randomisation   │  ← GTO balance
 *   │      └─ blocker / range-advantage      │
 *   └─────────────────────────────────────────┘
 *
 * Key improvements over v1:
 *   1. GTO mixed strategies (bluff/check-raise/slowplay at calibrated frequencies)
 *   2. Exploitative profile auto-selection based on villain VPIP/PFR
 *   3. Session-level learning: detects leaks, adjusts aggressiveness
 *   4. 4-bet and 5-bet/jam spots handled explicitly (fixes timeout bug)
 *   5. Pot-odds + MDF defends calculated on every street
 *   6. SPR-aware commitment logic
 *   7. Kelly-fraction bet sizing with session feedback
 *   8. Board texture and range advantage in bet sizing
 *   9. Blocker score used to select bluff candidates
 *  10. Postflop check-raises and overbets vs. capped ranges
 */

import { BotAction, defaultCheckAction, defaultFoldAction } from "../interfaces/ai-client-interfaces.ts";
import { Game } from "../models/game.ts";
import { Table } from "../models/table.ts";
import { PlayerAction } from "../models/player-action.ts";
import { PlayerStats } from "../models/player-stats.ts";

import { HandEvaluator, HandCategory } from "./hand-evaluator.ts";
import { EquityEstimator, EquityResult } from "./equity-estimator.ts";

import {
  StrategyProfile, StyleProfile,
  selectProfile, getProfile,
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
  countRangeCombos,
  assessRangeBalance,
} from "./hand-range-utils.ts";

import { SessionTracker, HandOutcome } from "./session-tracker.ts";

// ─── Internal Types ─────────────────────────────────────────────────────────

type PreflopSpot =
  | "unopened" | "limped" | "raised"
  | "reraised" | "four_bet" | "jammed";

type HandClass =
  | "air" | "draw" | "weakMade" | "strongMade" | "monster";

type Street = "preflop" | "flop" | "turn" | "river";

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
  facingRaiseSizeBB: number;
  numRaisesThisStreet: number;
  texture: BoardTexture;
  profile: StrategyProfile;
  mixSeed: string;        // deterministic seed for mixed-strategy decisions
};

export type DecisionTrace = {
  handKey: string;
  street: string;
  position: string;
  facingBet: boolean;
  spot?: string;
  handClass?: string;
  chosenAction: string;
  chosenSizeBB: number;
  reasons: string[];
  profile: string;
  equity?: number;
  potOdds?: number;
  spr?: number;
};

// ─── Constants ───────────────────────────────────────────────────────────────

const SPR_COMMITTED = 3;
const SPR_MEDIUM    = 8;

// Absolute premiums — always continue vs. any aggression
const ABSOLUTE_PREMIUMS = new Set(["AA","KK","QQ","AKs","AKo"]);
// Strong hands that call 3-bets / 4-bets in many spots
const STRONG_CALL_HANDS  = new Set(["JJ","TT","99","AQs","AQo","AJs","KQs"]);
// Hands that 4-bet/jam
const FOUR_BET_HANDS     = new Set(["AA","KK","QQ","JJ","AKs","AKo"]);
// Hands that call all-ins
const JAM_CALL_HANDS     = new Set(["AA","KK","QQ","JJ","TT","AKs","AQs","AKo"]);

// ─── Engine Class ─────────────────────────────────────────────────────────────

export class LocalPokerEngine {
  private readonly openRangeCache  = new Map<string, Set<string>>();
  private readonly isoRangeCache   = new Map<string, Set<string>>();

  private readonly handEvaluator   = new HandEvaluator();
  private readonly equityEstimator = new EquityEstimator();
  public  readonly sessionTracker  = new SessionTracker();

  private startingStackBB = 100;
  private lastTrace: DecisionTrace | null = null;

  // ── Public API ─────────────────────────────────────────────────────────────

  public decide(game: Game, table: Table): BotAction {
    const ctx = this.buildDecisionContext(game, table);
    if (!ctx) return defaultFoldAction;

    const action = ctx.street === "preflop"
      ? this.decidePreflopCtx(ctx)
      : this.decidePostflopCtx(ctx);

    return action;
  }

  public getLastTrace(): DecisionTrace | null { return this.lastTrace; }

  /** Call at session start to set bankroll baseline for Kelly sizing. */
  public setStartingStack(stackBB: number): void {
    this.startingStackBB = stackBB;
  }

  /**
   * Record the outcome of a hand.
   * Call this from bot.ts after each hand resolves.
   */
  public recordHandOutcome(outcome: HandOutcome): void {
    this.sessionTracker.recordHand(outcome);
  }

  public printSessionSummary(): void {
    this.sessionTracker.printSummary();
  }

  // ── Context Building ──────────────────────────────────────────────────────

  private buildDecisionContext(game: Game, table: Table): DecisionContext | null {
    const hero = game.getHero();
    if (!hero) return null;

    const hand = hero.getHand() ?? [];
    const heroId = hero.getPlayerId();
    const heroStackBB = hero.getStackSize();
    if (hand.length !== 2 || heroStackBB === undefined) return null;

    const board = this.getBoard(table);
    const actions = table.getPlayerActions();
    const rawStreet = (table.getStreet() || "preflop").toLowerCase();
    const street: Street =
      rawStreet === "flop" || rawStreet === "turn" || rawStreet === "river"
        ? rawStreet : "preflop";

    const heroPosition = table.getPlayerPositionFromId(heroId);
    const effectiveStackBB = this.getEffectiveStackBB(table, heroId, heroStackBB);
    const villain = this.getMainVillainProfile(table, heroId);

    // Select exploitative profile based on villain tendencies
    const villainVPIP  = villain?.vpip ?? 0;
    const villainPFR   = villain?.pfr ?? 0;
    const villainHands = villain?.totalHands ?? 0;
    const baseProfile  = selectProfile(villainVPIP, villainPFR, villainHands, effectiveStackBB);

    // Override with session-adapted profile (tilt-protection / leak-correction)
    const profile = this.sessionTracker.getAdaptiveProfile(
      baseProfile, heroStackBB, this.startingStackBB
    );

    const texture = analyzeBoardTexture(board);
    const facingRaiseSizeBB = this.getLastRaiseSizeBB(actions);
    const mixSeed = `${this.normalizeHand(hand)}:${street}:${board.join(",")}`;

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
      facingRaiseSizeBB,
      numRaisesThisStreet: this.countRaisesThisStreet(actions),
      texture,
      profile,
      mixSeed,
    };
  }

  // ── Preflop ───────────────────────────────────────────────────────────────

  private decidePreflopCtx(ctx: DecisionContext): BotAction {
    const {
      handKey, spot, heroPosition, numPlayers, actions,
      effectiveStackBB, villain, facingRaiseSizeBB, potBB, profile, mixSeed,
    } = ctx;

    const openRange    = this.getOpenRange(heroPosition, numPlayers);
    const isoRange     = this.getIsoRange(heroPosition, numPlayers, profile);
    const defendRange  = this.getDefendVsRaiseRange(heroPosition, effectiveStackBB, villain, profile);
    const jamCallRange = this.getCallVsJamRange(effectiveStackBB, villain, profile);
    const limpers      = this.countLimpers(actions);
    const lastRaise    = facingRaiseSizeBB || 2.5;
    const tier         = getPreflopTier(handKey);

    // ── Unopened ──────────────────────────────────────────────────────────
    if (spot === "unopened") {
      if (openRange.has(handKey)) {
        const size = this.getOpenSizeBB(heroPosition, numPlayers, effectiveStackBB);
        return this.emit(ctx, "raise", size, ["open-range", tier]);
      }
      // BB always gets to check their option
      if (heroPosition === "BB") return this.emit(ctx, "check", 0, ["bb-check-option"]);
      return this.emit(ctx, "fold", 0, ["outside-open-range", tier]);
    }

    // ── Limped ────────────────────────────────────────────────────────────
    if (spot === "limped") {
      if (isoRange.has(handKey)) {
        const size = this.getIsoSizeBB(limpers, heroPosition, effectiveStackBB, numPlayers);
        return this.emit(ctx, "raise", size, ["iso-raise", `limpers:${limpers}`, tier]);
      }
      // In position, consider over-limping speculative hands for value
      const canLimp = heroPosition === "BB" || heroPosition === "SB";
      if (canLimp && (tier === "speculative" || tier === "medium")) {
        return this.emit(ctx, "check", 0, ["limp-along-ip", tier]);
      }
      return this.emit(ctx, "fold", 0, ["iso-fold", tier]);
    }

    // ── Facing a single raise ─────────────────────────────────────────────
    if (spot === "raised") {
      // Always 3-bet absolute premiums
      if (ABSOLUTE_PREMIUMS.has(handKey)) {
        const size = this.getThreeBetSizeBB(lastRaise, heroPosition, effectiveStackBB);
        return this.emit(ctx, "raise", size, ["3bet-value-premium", tier]);
      }

      // Exploitative light 3-bets based on profile frequency
      if (profile.mixed.lightThreeBet > 0 && tier !== "marginal") {
        const lightSeed = mixSeed + ":light3b";
        if (
          mixedStrategyFires(profile.mixed.lightThreeBet, lightSeed) &&
          defendRange.has(handKey) &&
          this.isLightThreeBetCandidate(handKey, heroPosition, villain)
        ) {
          const size = this.getThreeBetSizeBB(lastRaise, heroPosition, effectiveStackBB);
          return this.emit(ctx, "raise", size, ["3bet-light-mixed", `freq:${profile.mixed.lightThreeBet}`, tier]);
        }
      }

      // Defend range call
      if (defendRange.has(handKey)) {
        const equity = this.equityEstimator.quickPreflopEquity(ctx.hand) ?? 0;
        const odds   = potOddsEquity(potBB, lastRaise);
        if (equity >= odds * 0.85) { // 15% implied-odds discount
          return this.emit(ctx, "call", 0, ["defend-call", `eq:${equity.toFixed(2)}`, tier]);
        }
      }

      return this.emit(ctx, "fold", 0, ["fold-to-raise", tier]);
    }

    // ── Facing a 3-bet (re-raise) ─────────────────────────────────────────
    // Every branch MUST return immediately — no fallthrough allowed.
    // This is the spot that caused the original timeout: the bot had no
    // decision path for certain hand+stack combinations and stalled.
    if (spot === "reraised") {

      // Step 1: 4-bet all premium hands — always, no conditions
      if (FOUR_BET_HANDS.has(handKey)) {
        // Short stack: just jam (avoids awkward SPR)
        if (effectiveStackBB <= 28) {
          return this.emit(ctx, "raise", effectiveStackBB,
            ["4bet-jam-premium", `eff:${effectiveStackBB.toFixed(1)}bb`]);
        }
        // Deep stack: size to ~2.5x the 3-bet
        const size = this.getFourBetSizeBB(lastRaise, effectiveStackBB);
        return this.emit(ctx, "raise", size, ["4bet-value", tier]);
      }

      // Step 2: Strong hands — evaluate calling vs folding immediately
      if (STRONG_CALL_HANDS.has(handKey)) {
        const equity = this.equityEstimator.quickPreflopEquity(ctx.hand) ?? 0;
        const odds   = potOddsEquity(potBB, lastRaise);
        const spr    = this.calcSPR(effectiveStackBB, potBB + lastRaise);

        // Pot-committed or equity clears the hurdle → call
        if (spr <= SPR_COMMITTED || equity >= odds * profile.impliedOddsFactor) {
          return this.emit(ctx, "call", 0,
            ["3bet-call-strong", `eq:${equity.toFixed(2)}`, `spr:${spr.toFixed(1)}`]);
        }
        // Doesn't clear the hurdle → fold immediately (no stalling)
        return this.emit(ctx, "fold", 0, ["3bet-fold-strong-bad-odds", tier]);
      }

      // Step 3: GTO bluff 4-bet — only for specific blocker hands, only OOP-safe stacks
      // This is a NARROW path — most hands skip straight to fold below
      if (
        effectiveStackBB <= 40 &&                                         // not too deep to bluff-jam
        this.is4BetBluffCandidate(handKey, heroPosition) &&
        mixedStrategyFires(profile.mixed.lightThreeBet * 0.5, mixSeed + ":bluff4b")
      ) {
        const size = this.getFourBetSizeBB(lastRaise, effectiveStackBB);
        return this.emit(ctx, "raise", size, ["4bet-bluff-mixed", tier]);
      }

      // Step 4: Everything else folds — fast, unconditional
      return this.emit(ctx, "fold", 0, ["fold-to-3bet", tier]);
    }

    // ── Facing a 4-bet ────────────────────────────────────────────────────
    if (spot === "four_bet") {
      if (ABSOLUTE_PREMIUMS.has(handKey) || (effectiveStackBB <= 20 && FOUR_BET_HANDS.has(handKey))) {
        return this.emit(ctx, "raise", effectiveStackBB, ["5bet-jam"]);
      }
      return this.emit(ctx, "fold", 0, ["fold-to-4bet", tier]);
    }

    // ── Jam facing ────────────────────────────────────────────────────────
    if (spot === "jammed") {
      if (jamCallRange.has(handKey)) {
        return this.emit(ctx, "call", 0, ["jam-call", tier]);
      }
      return this.emit(ctx, "fold", 0, ["fold-jam", tier]);
    }

    return defaultFoldAction;
  }

  // ── Postflop ──────────────────────────────────────────────────────────────

  private decidePostflopCtx(ctx: DecisionContext): BotAction {
    const { hand, board, numPlayers, effectiveStackBB, potBB,
            heroPosition, villain, facingBet, facingRaiseSizeBB,
            numRaisesThisStreet, street, texture, profile, mixSeed } = ctx;

    const handClass = this.classifyPostflopHand(hand, board, numPlayers);
    const madeHand  = this.evaluateMadeHand(hand, board);
    const equity    = this.estimateEquity(hand, board, Math.max(1, numPlayers - 1));
    const eqVal     = equity?.winRate ?? 0;

    const spr     = this.calcSPR(effectiveStackBB, potBB);
    const isRiver = street === "river";

    const rangeAdv = estimateRangeAdvantage(
      heroPosition, villain ? "BTN" : "CO", board, texture
    );
    const blockerScore = calcBlockerScore(hand, board);

    const baseReasons = [
      `hc:${madeHand?.category ?? "none"}`,
      `eq:${eqVal.toFixed(3)}`,
      `spr:${spr.toFixed(1)}`,
      `prof:${profile.name}`,
    ];

    // ── Not facing a bet: check or bet ─────────────────────────────────────
    if (!facingBet) {
      // Slowplay monsters at mixed frequency for balance
      if (handClass === "monster") {
        if (mixedStrategyFires(profile.mixed.slowplayMonster, mixSeed + ":slow")) {
          return this.emit(ctx, "check", 0, [...baseReasons, "slowplay-mixed"], handClass);
        }
        const size = this.getPostflopBetSize(potBB, effectiveStackBB, handClass, texture, profile, eqVal, spr);
        return this.emit(ctx, "bet", size, [...baseReasons, "monster-bet"], handClass);
      }

      // Check-raise setup: check a strong hand at mixed frequency
      if (handClass === "strongMade") {
        if (mixedStrategyFires(profile.mixed.checkRaiseStrong, mixSeed + ":xr")) {
          return this.emit(ctx, "check", 0, [...baseReasons, "xr-trap-mixed"], handClass);
        }
        if (eqVal >= profile.valueBetMinEquity) {
          const size = this.getPostflopBetSize(potBB, effectiveStackBB, handClass, texture, profile, eqVal, spr);
          return this.emit(ctx, "bet", size, [...baseReasons, "value-bet"], handClass);
        }
      }

      // Semi-bluff draws
      if (handClass === "draw" && !isRiver) {
        if (eqVal >= profile.semiBluffMinEquity) {
          // Prefer semi-bluffing with blockers (equity + blocker power)
          if (eqVal >= profile.semiBluffMinEquity || blockerScore >= 1) {
            if (mixedStrategyFires(
              handClass === "draw" ? profile.mixed.gutshotSemiBluff : 0.6,
              mixSeed + ":sbluff"
            )) {
              const size = this.getPostflopBetSize(potBB, effectiveStackBB, handClass, texture, profile, eqVal, spr);
              return this.emit(ctx, "bet", size, [...baseReasons, "semi-bluff", `blk:${blockerScore}`], handClass);
            }
          }
        }
      }

      // Protection bet: weak made hand but vulnerable on wet board
      if (handClass === "weakMade" && !isRiver && texture.wetScore >= 2 && eqVal >= 0.50) {
        const size = this.getPostflopBetSize(potBB, effectiveStackBB, "weakMade", texture, profile, eqVal, spr);
        return this.emit(ctx, "bet", size, [...baseReasons, "protection-bet"], handClass);
      }

      // Air c-bet on high-range-advantage boards at profile frequency
      if (handClass === "air" && !isRiver && rangeAdv > 0.10) {
        const cbFreq = street === "flop" ? profile.mixed.flopCBetAir
                     : street === "turn" ? profile.mixed.turnCBetAir
                     : profile.mixed.riverBluff;
        if (mixedStrategyFires(cbFreq, mixSeed + ":cbet")) {
          // Prefer bluffing hands with blockers to villain's nutted range
          if (blockerScore >= 1 || cbFreq >= 0.4) {
            const size = this.getPostflopBetSize(potBB, effectiveStackBB, "air", texture, profile, eqVal, spr);
            return this.emit(ctx, "bet", size, [...baseReasons, "range-cbet", `adv:${rangeAdv.toFixed(2)}`, `blk:${blockerScore}`], handClass);
          }
        }
      }

      // River bluff with blockers
      if (handClass === "air" && isRiver && blockerScore >= 2) {
        const villainFolds = estimateFoldEquity(
          villain?.vpip ?? 40, villain?.pfr ?? 20, villain?.totalHands ?? 0,
          0.75, texture
        );
        const betSize = potBB * 0.75;
        const ev = bluffEV(potBB, betSize, villainFolds);
        if (ev > 0 && mixedStrategyFires(profile.mixed.riverBluff, mixSeed + ":rbluff")) {
          return this.emit(ctx, "bet", betSize, [...baseReasons, `river-bluff-ev:${ev.toFixed(1)}`, `foldEq:${villainFolds.toFixed(2)}`], handClass);
        }
      }

      return this.emit(ctx, "check", 0, [...baseReasons, "check-default"], handClass);
    }

    // ── Facing a bet / raise ────────────────────────────────────────────────
    const potOdds  = potOddsEquity(potBB, facingRaiseSizeBB);
    const mdf      = minimumDefenceFrequency(potBB, facingRaiseSizeBB);
    const betFrac  = potBB > 0 ? facingRaiseSizeBB / potBB : 0.5;

    // Monster: check-raise or re-raise for value
    if (handClass === "monster") {
      if (numRaisesThisStreet < 2) {
        const raiseSize = this.getPostflopValueRaiseBB(potBB, facingRaiseSizeBB, effectiveStackBB, spr);
        return this.emit(ctx, "raise", raiseSize, [...baseReasons, "monster-raise"], handClass);
      }
      return this.emit(ctx, "call", 0, [...baseReasons, "monster-call-capped"], handClass);
    }

    // Strong made: call if equity justifies it; raise vs. passive villains
    if (handClass === "strongMade") {
      if (spr <= SPR_COMMITTED) {
        return this.emit(ctx, "call", 0, [...baseReasons, "strong-committed"], handClass);
      }
      if (eqVal >= potOdds * profile.impliedOddsFactor) {
        // Check-raise with strong hands if we trapped earlier
        if (numRaisesThisStreet === 0 && mixedStrategyFires(profile.mixed.checkRaiseStrong * 0.5, mixSeed + ":xr2")) {
          const raiseSize = this.getPostflopValueRaiseBB(potBB, facingRaiseSizeBB, effectiveStackBB, spr);
          return this.emit(ctx, "raise", raiseSize, [...baseReasons, "xr-value", `eq:${eqVal.toFixed(2)}`], handClass);
        }
        return this.emit(ctx, "call", 0, [...baseReasons, "strong-call", `eq:${eqVal.toFixed(2)}`], handClass);
      }
      return this.emit(ctx, "fold", 0, [...baseReasons, "strong-fold-bad-odds", `eq:${eqVal.toFixed(2)}`], handClass);
    }

    // Draw: call if implied odds justify it
    if (handClass === "draw" && !isRiver) {
      const impliedOddsAdj = potOdds * profile.impliedOddsFactor;
      if (eqVal >= impliedOddsAdj || spr <= SPR_COMMITTED) {
        return this.emit(ctx, "call", 0, [...baseReasons, "draw-call", `impl:${impliedOddsAdj.toFixed(2)}`], handClass);
      }
      // Semi-bluff raise on draws with blockers vs. passive villain
      if (
        eqVal >= 0.30 &&
        blockerScore >= 1 &&
        numRaisesThisStreet < 2 &&
        villain && villain.vpip > 40 && villain.pfr < 20
      ) {
        const raiseSize = this.getPostflopValueRaiseBB(potBB, facingRaiseSizeBB, effectiveStackBB, spr);
        return this.emit(ctx, "raise", raiseSize, [...baseReasons, "draw-semibluff-raise", `blk:${blockerScore}`], handClass);
      }
    }

    // Weak made hand: BB defence
    if (handClass === "weakMade" && heroPosition === "BB" && eqVal >= potOdds) {
      return this.emit(ctx, "call", 0, [...baseReasons, "bb-defend", `eq:${eqVal.toFixed(2)}`], handClass);
    }

    // Pot-committed regardless
    if (spr <= SPR_COMMITTED) {
      return this.emit(ctx, "call", 0, [...baseReasons, "committed-spr"], handClass);
    }

    // MDF-based defence: defend enough hands to prevent villain from profitably betting any two cards
    // Check if we're above the MDF threshold in our overall range
    // (Simplified: defend if equity > MDF-adjusted threshold)
    if (eqVal >= profile.callEquityThreshold) {
      return this.emit(ctx, "call", 0, [...baseReasons, "mdf-defend", `mdf:${mdf.toFixed(2)}`], handClass);
    }

    return this.emit(ctx, "fold", 0, [...baseReasons, "fold-insufficient"], handClass);
  }

  // ── Spot Classification ────────────────────────────────────────────────────

  private classifyPreflopSpot(actions: PlayerAction[]): PreflopSpot {
    const voluntary = actions.filter((a) => a.getAction().toLowerCase() !== "post");
    const raises = voluntary.filter((a) => {
      const act = a.getAction().toLowerCase();
      return act === "raise" || act === "bet";
    });
    const calls = voluntary.filter((a) => a.getAction().toLowerCase() === "call");

    if (raises.length === 0 && calls.length === 0) return "unopened";
    if (raises.length === 0 && calls.length > 0) return "limped";

    const biggestRaise = Math.max(...raises.map((a) => a.getBetAmount()), 0);
    if (biggestRaise >= 10) return "jammed";
    if (raises.length >= 3) return "four_bet";
    if (raises.length >= 2) return "reraised";
    return "raised";
  }

  // ── Sizing Logic ───────────────────────────────────────────────────────────

  private getOpenSizeBB(position: string, numPlayers: number, effectiveStackBB: number): number {
    if (effectiveStackBB <= 12) return effectiveStackBB;
    if (numPlayers === 2 && position === "SB") return Math.min(2.5, effectiveStackBB);
    const sizes: Record<string, number> = {
      SB: 3.0, BU: 2.3, CO: 2.3,
      HJ: 2.5, LJ: 2.5, MP: 2.5, "UTG+1": 2.5, UTG: 2.5,
    };
    return Math.min(sizes[position] ?? 2.5, effectiveStackBB);
  }

  private getIsoSizeBB(
    limpers: number, heroPosition: string, effectiveStackBB: number, numPlayers: number
  ): number {
    if (effectiveStackBB <= 12) return effectiveStackBB;
    let size = 3.5 + Math.max(0, limpers - 1);
    if (heroPosition === "BB" || heroPosition === "SB") size += 0.5;
    if (numPlayers === 2 && heroPosition === "BB") size = 4.0;
    return Math.min(size, effectiveStackBB);
  }

  private getThreeBetSizeBB(lastRaiseBB: number, heroPosition: string, effectiveStackBB: number): number {
    if (effectiveStackBB <= 20) return effectiveStackBB;
    const inPos = ["BU","CO","HJ"].includes(heroPosition);
    const multiplier = inPos ? 3.0 : 4.0;
    return Math.min(Math.max(lastRaiseBB * multiplier, 7.5), effectiveStackBB);
  }

  private getFourBetSizeBB(threeBetBB: number, effectiveStackBB: number): number {
    if (effectiveStackBB <= 25) return effectiveStackBB;
    return Math.min(Math.max(threeBetBB * 2.5, 18), effectiveStackBB);
  }

  private getPostflopBetSize(
    potBB: number, effectiveStackBB: number, handClass: HandClass,
    texture: BoardTexture, profile: StrategyProfile, equity: number, spr: number
  ): number {
    if (effectiveStackBB <= 10) return effectiveStackBB;

    // Low SPR: jam the remaining stack
    if (spr <= 2.5) return Math.min(effectiveStackBB, potBB);

    // Select sizing fraction from profile based on hand strength and board texture
    let fraction: number;
    if (handClass === "monster") {
      // Wet board: bet larger to charge draws; dry board: smaller to induce
      fraction = texture.wetScore >= 3 ? profile.sizing.overbetFraction
               : texture.wetScore >= 2 ? profile.sizing.largeFraction
               : profile.sizing.medFraction;
    } else if (handClass === "strongMade") {
      fraction = texture.wetScore >= 2 ? profile.sizing.largeFraction : profile.sizing.medFraction;
    } else if (handClass === "weakMade") {
      fraction = profile.sizing.smallFraction;
    } else {
      // draw / air / bluff
      fraction = texture.wetScore >= 2 ? profile.sizing.medFraction : profile.sizing.smallFraction;
    }

    // Kelly adjustment from session tracker
    const kellyAdj = this.sessionTracker.kellyBetAdjustment(
      equity, potBB, potBB * fraction, profile.kellyFraction
    );
    const size = potBB * fraction * kellyAdj;
    return Math.min(Math.max(size, 1), effectiveStackBB);
  }

  private getPostflopValueRaiseBB(
    potBB: number, betFacingBB: number, effectiveStackBB: number, spr: number
  ): number {
    if (effectiveStackBB <= 10 || spr <= 2) return effectiveStackBB;
    const size = Math.max(betFacingBB * 2.8, potBB * 0.9);
    return Math.min(size, effectiveStackBB);
  }

  // ── Range Construction ────────────────────────────────────────────────────

  private getOpenRange(position: string, numPlayers: number): Set<string> {
    const key = `${position}:${numPlayers}`;
    const cached = this.openRangeCache.get(key);
    if (cached) return cached;

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

    let range: Set<string>;
    if (numPlayers === 2 && position === "SB") {
      range = new Set([
        "AA","KK","QQ","JJ","TT","99","88","77","66","55",
        "AKs","AQs","AJs","ATs","A9s","A8s","A7s","A6s","A5s","A4s","A3s","A2s",
        "KQs","KJs","KTs","QJs","QTs","JTs","T9s","98s","87s","76s",
        "AKo","AQo","AJo","ATo","KQo","KJo","QJo","JTo",
      ]);
    } else {
      range = new Set(ranges[position] ?? []);
    }

    this.openRangeCache.set(key, range);
    return range;
  }

  private getIsoRange(position: string, numPlayers: number, profile: StrategyProfile): Set<string> {
    const key = `${position}:${numPlayers}:${profile.name}`;
    const cached = this.isoRangeCache.get(key);
    if (cached) return cached;

    const base  = this.getOpenRange(position, numPlayers);
    // Loose / LAG profiles add more iso hands
    const extra = profile.name === "loose" || profile.name === "lag"
      ? ["KTo","QTo","97s","86s","75s","54s","A7o","A6o","K9o","Q9o","J9o","T8s"]
      : ["KTo","QTo","97s","86s","75s","54s"];

    const range = new Set([...base, ...extra]);
    this.isoRangeCache.set(key, range);
    return range;
  }

  private getDefendVsRaiseRange(
    position: string, effectiveStackBB: number,
    villain: VillainProfile | null, profile: StrategyProfile
  ): Set<string> {
    let range: Set<string>;

    if (position === "BB") {
      range = new Set([
        "AA","KK","QQ","JJ","TT","99","88","77","66",
        "AKs","AQs","AJs","ATs","A9s","KQs","KJs","QJs","JTs","T9s","98s",
        "AKo","AQo","AJo","KQo","QJo",
      ]);
    } else if (["SB","BU","CO"].includes(position)) {
      range = new Set([
        "AA","KK","QQ","JJ","TT","99","88",
        "AKs","AQs","AJs","ATs","KQs","QJs","JTs",
        "AKo","AQo","AJo","KQo",
      ]);
    } else {
      range = new Set(["AA","KK","QQ","JJ","TT","99","AKs","AQs","AJs","KQs","AKo","AQo"]);
    }

    // Short stack — narrow to only strong hands
    if (effectiveStackBB <= 20) {
      range = new Set([...range].filter((h) =>
        ["AA","KK","QQ","JJ","TT","99","AKs","AQs","AKo","AQo"].includes(h)
      ));
    }

    // Wider vs. loose raisers
    if (villain && villain.totalHands >= 20 && villain.vpip > 45 && villain.pfr > 25) {
      ["A9s","KTs","QTs","J9s","T8s","ATo","KJo"].forEach((h) => range.add(h));
    }

    // Profile-based expansion (loose/lag defend more)
    if (profile.name === "loose" || profile.name === "lag") {
      ["A8s","A7s","K9s","Q9s","J9s","87s"].forEach((h) => range.add(h));
    }

    return range;
  }

  private getCallVsJamRange(
    effectiveStackBB: number, villain: VillainProfile | null, profile: StrategyProfile
  ): Set<string> {
    const range = new Set<string>(["AA","KK","QQ","JJ","TT","AKs","AQs","AKo"]);
    if (effectiveStackBB <= 15) ["99","88","AQo","AJs"].forEach((h) => range.add(h));
    if (effectiveStackBB <= 10) ["77","ATs","KQs","AJo"].forEach((h) => range.add(h));
    if (villain && villain.totalHands >= 20 && villain.vpip > 50 && villain.pfr > 35) {
      ["99","88","AQo","AJs","ATs"].forEach((h) => range.add(h));
    }
    if (profile.name === "loose" || profile.name === "lag") {
      range.add("99"); range.add("AQo");
    }
    return range;
  }

  // ── Light 3-bet / 4-bet Candidate Selection ──────────────────────────────

  private isLightThreeBetCandidate(
    handKey: string, heroPosition: string, villain: VillainProfile | null
  ): boolean {
    const inPosition = ["BU","CO","HJ"].includes(heroPosition);
    if (!inPosition) return false;
    // Use suited connectors and suited aces as 3-bet bluffs
    const lightHands = new Set([
      "AJs","A5s","A4s","KQs","KJs","QJs","JTs","T9s","98s",
      "87s","76s","65s",
    ]);
    if (!lightHands.has(handKey)) return false;
    // More effective vs. wide openers
    if (villain && villain.totalHands >= 15 && villain.pfr > 22) return true;
    return inPosition; // Default: use position
  }

  private is4BetBluffCandidate(handKey: string, heroPosition: string): boolean {
    // 4-bet bluff with hands that block villain's strong combos
    // A-high blocks AA/AK, K-high blocks KK
    const blockerHands = new Set(["AJs","A5s","A4s","A3s","KQs","KJs"]);
    return blockerHands.has(handKey) && ["BU","CO"].includes(heroPosition);
  }

  // ── Hand Classification ────────────────────────────────────────────────────

  private classifyPostflopHand(hand: string[], board: string[], numPlayers: number): HandClass {
    const madeHand = this.evaluateMadeHand(hand, board);
    const equity   = this.estimateEquity(hand, board, Math.max(1, numPlayers - 1));

    if (madeHand) return this.mapMadeHandToClass(madeHand.category, equity, hand, board);

    if (this.hasFlushDraw(hand, board))              return "draw";
    if (this.hasOpenEndedStraightDraw(hand, board)) return "draw";
    if (this.hasGutshotStraightDraw(hand, board))   return "draw";

    return "air";
  }

  private evaluateMadeHand(hand: string[], board: string[]) {
    if (board.length < 3) return null;
    return this.handEvaluator.evaluate([...hand, ...board]);
  }

  private estimateEquity(hand: string[], board: string[], numOpponents: number): EquityResult | null {
    if (board.length < 3) return null;
    const trials = board.length >= 5 ? 400 : board.length >= 4 ? 600 : 800;
    return this.equityEstimator.estimateVsRandomRange(hand, board, Math.max(1, numOpponents), trials);
  }

  private mapMadeHandToClass(
    category: HandCategory, equity: EquityResult | null,
    hand: string[], board: string[]
  ): HandClass {
    if (["straight-flush","four-of-a-kind","full-house"].includes(category)) return "monster";
    if (["flush","straight","three-of-a-kind","two-pair"].includes(category)) return "strongMade";

    if (category === "pair") {
      if (this.isTopPairOrOverpair(hand, board)) return "strongMade";
      if (equity && equity.winRate >= 0.55) return "strongMade";
      return "weakMade";
    }

    if (equity && equity.winRate >= 0.28 && board.length < 5) return "draw";
    return "air";
  }

  // ── Utility ───────────────────────────────────────────────────────────────

  private calcSPR(effectiveStackBB: number, potBB: number): number {
    return potBB > 0 ? effectiveStackBB / potBB : 999;
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
    const raises = actions.filter((a) => {
      const act = a.getAction().toLowerCase();
      return act === "raise" || act === "bet";
    });
    if (raises.length > 0) return 0;
    return actions.filter((a) => a.getAction().toLowerCase() === "call").length;
  }

  private getLastRaiseSizeBB(actions: PlayerAction[]): number {
    const raises = actions.filter((a) => {
      const act = a.getAction().toLowerCase();
      return act === "raise" || act === "bet";
    });
    return raises.length > 0 ? raises[raises.length - 1].getBetAmount() : 0;
  }

  private getBoard(table: Table): string[] {
    const runout = table.getRunout();
    if (!runout) return [];
    return runout.replace(/[\[\]]/g, "").split(",").map((s) => s.trim()).filter(Boolean);
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
      } catch { continue; }
    }
    return null;
  }

  private toVillainProfile(stats: PlayerStats): VillainProfile {
    const totalHands = stats.getTotalHands();
    const vpip = totalHands > 0 ? (stats.getVPIPHands() / totalHands) * 100 : 0;
    const pfr  = totalHands > 0 ? (stats.getPFRHands()  / totalHands) * 100 : 0;
    return { totalHands, vpip, pfr };
  }

  private normalizeHand(cards: string[]): string {
    const order = "AKQJT98765432";
    const [a, b] = cards;
    const r1 = a[0].toUpperCase(), r2 = b[0].toUpperCase();
    const [hi, lo] = [r1, r2].sort((x, y) => order.indexOf(x) - order.indexOf(y));
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
    const boardTop   = this.getBoardTopRank(board);
    const holeRanks  = hand.map((c) => c[0].toUpperCase());
    if (holeRanks.includes(boardTop)) return true;
    if (holeRanks[0] === holeRanks[1]) {
      const order = "AKQJT98765432";
      return order.indexOf(holeRanks[0]) < order.indexOf(boardTop);
    }
    return false;
  }

  private hasFlushDraw(hand: string[], board: string[]): boolean {
    const suits = new Map<string, number>();
    for (const card of [...hand, ...board]) {
      const s = card[1];
      suits.set(s, (suits.get(s) ?? 0) + 1);
    }
    return [...suits.values()].some((v) => v === 4);
  }

  private getUniqueSortedRanks(cards: string[]): number[] {
    const map: Record<string, number> = {
      A:14,K:13,Q:12,J:11,T:10,"9":9,"8":8,"7":7,"6":6,"5":5,"4":4,"3":3,"2":2,
    };
    const vals = new Set<number>();
    for (const c of cards) {
      const r = c[0].toUpperCase();
      const v = map[r];
      if (v) { vals.add(v); if (r === "A") vals.add(1); }
    }
    return [...vals].sort((a, b) => a - b);
  }

  private hasOpenEndedStraightDraw(hand: string[], board: string[]): boolean {
    const values = this.getUniqueSortedRanks([...hand, ...board]);
    for (let i = 0; i <= values.length - 4; i++) {
      if (values[i + 3] - values[i] === 3) return true;
    }
    return false;
  }

  private hasGutshotStraightDraw(hand: string[], board: string[]): boolean {
    const values = this.getUniqueSortedRanks([...hand, ...board]);
    for (let i = 0; i <= values.length - 4; i++) {
      if (values[i + 3] - values[i] === 4) return true;
    }
    return false;
  }

  // ── Emit helper (sets trace + returns action) ─────────────────────────────

  private emit(
    ctx: DecisionContext,
    action: string,
    sizeBB: number,
    reasons: string[],
    handClass?: HandClass
  ): BotAction {
    const equity = ctx.board.length >= 3
      ? this.estimateEquity(ctx.hand, ctx.board, Math.max(1, ctx.numPlayers - 1))?.winRate
      : this.equityEstimator.quickPreflopEquity(ctx.hand) ?? undefined;

    const potOdds = ctx.facingBet && ctx.facingRaiseSizeBB > 0
      ? potOddsEquity(ctx.potBB, ctx.facingRaiseSizeBB)
      : undefined;

    const spr = ctx.potBB > 0 ? ctx.effectiveStackBB / ctx.potBB : undefined;

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
      potOdds,
      spr,
    };

    if (action === "fold")  return defaultFoldAction;
    if (action === "check") return defaultCheckAction;
    return { action_str: action as any, bet_size_in_BBs: sizeBB };
  }
}