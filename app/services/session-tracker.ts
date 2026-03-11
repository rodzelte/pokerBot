/**
 * session-tracker.ts
 *
 * Tracks hand outcomes and persists learning to SQLite so the bot improves
 * across sessions — not just within a single run.
 *
 * Persistence model:
 *   Table: BotSession       — one row per session (aggregate stats)
 *   Table: BotHandHistory   — one row per hand (full outcome log)
 *   Table: BotLeakCounters  — lifetime rolling counters (survives restarts)
 *
 * On startup: loadLifetimeStats() is called to seed in-memory counters from DB.
 * After each hand: saveHandToDB() appends the outcome and updates counters.
 * On shutdown / summary: finaliseSession() writes the session aggregate row.
 *
 * This means the bot's strategy bias, tilt detection, and bluff thresholds are
 * informed by ALL hands ever played — not just the current session.
 */

import { StyleProfile, selectProfile, getProfile } from "./gto-strategy-profiles.ts";
import type { StrategyProfile } from "./gto-strategy-profiles.ts";
import db_service from "../services/db-service.ts";

// ─── Types ────────────────────────────────────────────────────────────────

export type HandOutcome = {
  handId: string;
  street: string;
  heroPosition: string;
  handKey: string;
  action: string;
  sizeBB: number;
  potBB: number;
  wonBB: number;
  showdown: boolean;
  villainVPIP?: number;
  villainPFR?: number;
  notes?: string;
};

export type SessionStats = {
  handsPlayed: number;
  handsWon: number;
  totalWonBB: number;
  bbPer100: number;
  biggestWin: number;
  biggestLoss: number;
  showdownWinRate: number;
  nonShowdownWinRate: number;
  callAccuracy: number;
  bluffSuccessRate: number;
  threeBetSuccessRate: number;
};

export type LeakReport = {
  description: string;
  severity: "low" | "medium" | "high";
  recommendedAdjustment: string;
};

// ─── LifetimeStats loaded from DB ────────────────────────────────────────

type LifetimeCounters = {
  totalHands: number;
  totalWonBB: number;
  callTotal: number;
  callWins: number;
  bluffTotal: number;
  bluffSuccess: number;
  threeBetTotal: number;
  threeBetWins: number;
  showdownTotal: number;
  showdownWins: number;
};

// ─── SessionTracker ────────────────────────────────────────────────────────

export class SessionTracker {
  private history: HandOutcome[] = [];
  private sessionId: string;
  private sessionStart = Date.now();
  private dbReady = false;

  // Session-only rolling counters
  private netBB = 0;
  private showdownWins = 0;
  private showdownTotal = 0;
  private nonShowdownWins = 0;
  private nonShowdownTotal = 0;
  private callWins = 0;
  private callTotal = 0;
  private bluffSuccess = 0;
  private bluffTotal = 0;
  private threeBetWins = 0;
  private threeBetTotal = 0;
  private consecutiveLosses = 0;
  private consecutiveWins = 0;
  private currentBias: StyleProfile = "gto";
  private villainAdj: {
    callThresholdDelta: number;
    bluffFreqDelta: number;
    threeBetFreqDelta: number;
    foldToCBet: boolean;
  } | null = null;

  // Lifetime counters — loaded from DB on init
  private lifetime: LifetimeCounters = {
    totalHands: 0, totalWonBB: 0,
    callTotal: 0, callWins: 0,
    bluffTotal: 0, bluffSuccess: 0,
    threeBetTotal: 0, threeBetWins: 0,
    showdownTotal: 0, showdownWins: 0,
  };

  constructor() {
    this.sessionId = `session_${Date.now()}`;
  }

  // ── Initialisation ──────────────────────────────────────────────────────

  /**
   * MUST be called once at bot startup (after db_service.init()).
   * Creates DB tables if they don't exist, then loads lifetime stats.
   */
  public async init(): Promise<void> {
    try {
      await this.createTables();
      await this.loadLifetimeStats();
      this.dbReady = true;
      console.log(
        `[SessionTracker] Loaded lifetime data: ` +
        `${this.lifetime.totalHands} hands, ` +
        `${this.lifetimeBBPer100().toFixed(1)} BB/100 all-time`
      );
    } catch (err) {
      console.log("[SessionTracker] DB init failed — running in-memory only:", err);
    }
  }

  // ── Recording ───────────────────────────────────────────────────────────

  public recordHand(outcome: HandOutcome): void {
    this.history.push(outcome);
    this.netBB += outcome.wonBB;

    if (outcome.wonBB > 0) {
      this.consecutiveWins++;
      this.consecutiveLosses = 0;
    } else if (outcome.wonBB < 0) {
      this.consecutiveLosses++;
      this.consecutiveWins = 0;
    }

    if (outcome.showdown) {
      this.showdownTotal++;
      this.lifetime.showdownTotal++;
      if (outcome.wonBB > 0) {
        this.showdownWins++;
        this.lifetime.showdownWins++;
      }
    } else if (outcome.action !== "fold" && outcome.wonBB > 0) {
      this.nonShowdownTotal++;
      this.nonShowdownWins++;
    } else if (outcome.action !== "fold") {
      this.nonShowdownTotal++;
    }

    if (outcome.action === "call") {
      this.callTotal++;
      this.lifetime.callTotal++;
      if (outcome.wonBB > 0) {
        this.callWins++;
        this.lifetime.callWins++;
      }
    }

    const isBet = outcome.action === "bet" || outcome.action === "raise";
    if (isBet && !outcome.showdown) {
      this.bluffTotal++;
      this.lifetime.bluffTotal++;
      if (outcome.wonBB > 0) {
        this.bluffSuccess++;
        this.lifetime.bluffSuccess++;
      }
    }

    if (outcome.action === "raise" && outcome.street === "preflop") {
      this.threeBetTotal++;
      this.lifetime.threeBetTotal++;
      if (outcome.wonBB > 0) {
        this.threeBetWins++;
        this.lifetime.threeBetWins++;
      }
    }

    this.lifetime.totalHands++;
    this.lifetime.totalWonBB += outcome.wonBB;

    this.updateBias();

    // Persist asynchronously — do NOT await (keeps decide() synchronous)
    if (this.dbReady) {
      this.saveHandToDB(outcome).catch((err) =>
        console.log("[SessionTracker] Failed to save hand:", err)
      );
    }
  }

  // ── Stats & Leaks ────────────────────────────────────────────────────────

  public getSessionStats(): SessionStats {
    const hands = this.history.length;
    const won   = this.history.filter((h) => h.wonBB > 0).length;
    const best  = this.history.reduce((m, h) => Math.max(m, h.wonBB), 0);
    const worst = this.history.reduce((m, h) => Math.min(m, h.wonBB), 0);

    return {
      handsPlayed: hands,
      handsWon: won,
      totalWonBB: this.netBB,
      bbPer100: hands > 0 ? (this.netBB / hands) * 100 : 0,
      biggestWin: best,
      biggestLoss: worst,
      showdownWinRate: this.showdownTotal > 0 ? this.showdownWins / this.showdownTotal : 0,
      nonShowdownWinRate: this.nonShowdownTotal > 0 ? this.nonShowdownWins / this.nonShowdownTotal : 0,
      callAccuracy: this.callTotal > 0 ? this.callWins / this.callTotal : 0,
      bluffSuccessRate: this.bluffTotal > 0 ? this.bluffSuccess / this.bluffTotal : 0,
      threeBetSuccessRate: this.threeBetTotal > 0 ? this.threeBetWins / this.threeBetTotal : 0,
    };
  }

  public detectLeaks(): LeakReport[] {
    const leaks: LeakReport[] = [];

    // Use lifetime counters when we have enough data; fall back to session
    const useLifetime = this.lifetime.totalHands >= 50;
    const calls  = useLifetime ? this.lifetime.callTotal  : this.callTotal;
    const cWins  = useLifetime ? this.lifetime.callWins   : this.callWins;
    const bluffs = useLifetime ? this.lifetime.bluffTotal : this.bluffTotal;
    const bWins  = useLifetime ? this.lifetime.bluffSuccess : this.bluffSuccess;
    const hands  = useLifetime ? this.lifetime.totalHands : this.history.length;

    if (hands < 20) return leaks;

    const callAcc  = calls  > 8  ? cWins / calls  : null;
    const bluffAcc = bluffs > 8  ? bWins / bluffs : null;

    if (callAcc !== null && callAcc < 0.35) {
      leaks.push({
        description: `Call accuracy ${(callAcc * 100).toFixed(1)}% (threshold 35%)${useLifetime ? " — lifetime" : ""}`,
        severity: "high",
        recommendedAdjustment: "Tighten calling ranges — only continue with top of range",
      });
    }

    if (bluffAcc !== null && bluffAcc < 0.38) {
      leaks.push({
        description: `Bluff success ${(bluffAcc * 100).toFixed(1)}% (threshold 38%)${useLifetime ? " — lifetime" : ""}`,
        severity: "medium",
        recommendedAdjustment: "Reduce bluff frequency — villain is calling down too wide",
      });
    }

    if (hands >= 30 && bluffs === 0) {
      leaks.push({
        description: "Zero bluffs recorded — range is completely unbalanced",
        severity: "low",
        recommendedAdjustment: "Add semi-bluffs on draws and c-bets to balance range",
      });
    }

    const stats = this.getSessionStats();
    if (stats.handsPlayed >= 20 && this.nonShowdownTotal >= 10 && stats.nonShowdownWinRate < 0.30) {
      leaks.push({
        description: `Non-showdown win rate ${(stats.nonShowdownWinRate * 100).toFixed(1)}% — fold equity too low`,
        severity: "medium",
        recommendedAdjustment: "Use larger bet sizes — villain is not folding to small bets",
      });
    }

    const lifetimeBB100 = this.lifetimeBBPer100();
    if (this.lifetime.totalHands >= 50 && lifetimeBB100 < -10) {
      leaks.push({
        description: `Lifetime BB/100 is ${lifetimeBB100.toFixed(1)} — structural losing pattern`,
        severity: "high",
        recommendedAdjustment: "Switch to TIGHT profile — reduce variance and wait for premium spots",
      });
    }

    return leaks;
  }

  public getAdaptiveProfile(
    base: StrategyProfile,
    currentStackBB: number,
    startingStackBB: number
  ): StrategyProfile {
    let profile = base;
    const stackRatio = startingStackBB > 0 ? currentStackBB / startingStackBB : 1;

    if (stackRatio < 0.70) {
      profile = {
        ...profile,
        callEquityThreshold: Math.min(0.50, profile.callEquityThreshold + 0.04),
      };
    }

    if (stackRatio > 1.30) {
      profile = {
        ...profile,
        callEquityThreshold: Math.max(0.22, profile.callEquityThreshold - 0.04),
      };
    }

    if (this.villainAdj) {
      const adj = this.villainAdj;
      profile = {
        ...profile,
        callEquityThreshold: Math.max(
          0.15,
          Math.min(0.60, profile.callEquityThreshold + adj.callThresholdDelta)
        ),
        mixed: {
          ...profile.mixed,
          flopCBetAir: Math.max(0, Math.min(0.90, profile.mixed.flopCBetAir + adj.bluffFreqDelta)),
          turnCBetAir: Math.max(0, Math.min(0.90, profile.mixed.turnCBetAir + adj.bluffFreqDelta * 0.70)),
          riverBluff: Math.max(0, Math.min(0.80, profile.mixed.riverBluff + adj.bluffFreqDelta * 0.50)),
          lightThreeBet: Math.max(0, Math.min(0.70, profile.mixed.lightThreeBet + adj.threeBetFreqDelta)),
        },
      };
    }

    return profile;
  }

  public applyVillainAdjustments(adj: {
    callThresholdDelta: number;
    bluffFreqDelta: number;
    threeBetFreqDelta: number;
    foldToCBet: boolean;
  }): void {
    this.villainAdj = adj;
  }

  public clearVillainAdjustments(): void {
    this.villainAdj = null;
  }

  public kellyBetAdjustment(
    equity: number, potBB: number, baseBetBB: number, kellyFraction: number
  ): number {
    if (baseBetBB <= 0 || potBB <= 0) return 1.0;
    const b = potBB / baseBetBB;
    const p = Math.min(0.98, Math.max(0.02, equity));
    const q = 1 - p;
    const fullKelly = (b * p - q) / b;
    return Math.max(0.60, Math.min(1.40, 1 + fullKelly * kellyFraction));
  }

  public getConsecutiveLosses(): number { return this.consecutiveLosses; }
  public getNetBB(): number { return this.netBB; }
  public getHandCount(): number { return this.history.length; }
  public getLifetimeHands(): number { return this.lifetime.totalHands; }
  public exportHistory(): HandOutcome[] { return [...this.history]; }

  public async finaliseSession(): Promise<void> {
    if (!this.dbReady) return;
    const stats = this.getSessionStats();
    try {
      await db_service.query(
        `INSERT OR REPLACE INTO BotSession
           (session_id, started_at, hands_played, net_bb, bb_per_100,
            sd_win_rate, call_accuracy, bluff_success)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
        [
          this.sessionId,
          new Date(this.sessionStart).toISOString(),
          stats.handsPlayed,
          stats.totalWonBB,
          stats.bbPer100,
          stats.showdownWinRate,
          stats.callAccuracy,
          stats.bluffSuccessRate,
        ]
      );
      await this.saveLifetimeCounters();
      console.log("[SessionTracker] Session saved to DB.");
    } catch (err) {
      console.log("[SessionTracker] Failed to finalise session:", err);
    }
  }

  public printSummary(): void {
    const s = this.getSessionStats();
    const leaks = this.detectLeaks();
    const minutes = Math.round((Date.now() - this.sessionStart) / 60000);

    console.log("\n════════ SESSION SUMMARY ════════");
    console.log(`  Session ID:      ${this.sessionId}`);
    console.log(`  Duration:        ${minutes} min`);
    console.log(`  Hands (session): ${s.handsPlayed}`);
    console.log(`  Hands (lifetime):${this.lifetime.totalHands}`);
    console.log(`  Net (session):   ${s.totalWonBB >= 0 ? "+" : ""}${s.totalWonBB.toFixed(1)} BB`);
    console.log(`  BB/100 (session):${s.bbPer100.toFixed(1)}`);
    console.log(`  BB/100 (lifetime):${this.lifetimeBBPer100().toFixed(1)}`);
    console.log(`  SD win rate:     ${(s.showdownWinRate * 100).toFixed(1)}%`);
    console.log(`  Call accuracy:   ${(s.callAccuracy * 100).toFixed(1)}%`);
    console.log(`  Bluff success:   ${(s.bluffSuccessRate * 100).toFixed(1)}%`);
    console.log(`  Strategy bias:   ${this.currentBias}`);

    if (leaks.length > 0) {
      console.log("\n  ⚠ Leaks detected:");
      for (const l of leaks) {
        console.log(`    [${l.severity.toUpperCase()}] ${l.description}`);
        console.log(`      → ${l.recommendedAdjustment}`);
      }
    } else {
      console.log("  ✓ No significant leaks detected");
    }
    console.log("═════════════════════════════════\n");
  }

  // ── Private DB helpers ───────────────────────────────────────────────────

  private async createTables(): Promise<void> {
    // Session aggregate table
    await db_service.query(
      `CREATE TABLE IF NOT EXISTS BotSession (
         session_id    TEXT PRIMARY KEY,
         started_at    TEXT NOT NULL,
         hands_played  INTEGER NOT NULL DEFAULT 0,
         net_bb        REAL    NOT NULL DEFAULT 0,
         bb_per_100    REAL    NOT NULL DEFAULT 0,
         sd_win_rate   REAL    NOT NULL DEFAULT 0,
         call_accuracy REAL    NOT NULL DEFAULT 0,
         bluff_success REAL    NOT NULL DEFAULT 0
       )`, []
    );

    // Per-hand history (full log for future analysis)
    await db_service.query(
      `CREATE TABLE IF NOT EXISTS BotHandHistory (
         id            INTEGER PRIMARY KEY AUTOINCREMENT,
         session_id    TEXT    NOT NULL,
         hand_id       TEXT    NOT NULL,
         street        TEXT    NOT NULL,
         position      TEXT    NOT NULL,
         hand_key      TEXT    NOT NULL,
         action        TEXT    NOT NULL,
         size_bb       REAL    NOT NULL DEFAULT 0,
         pot_bb        REAL    NOT NULL DEFAULT 0,
         won_bb        REAL    NOT NULL DEFAULT 0,
         showdown      INTEGER NOT NULL DEFAULT 0,
         villain_vpip  REAL,
         villain_pfr   REAL,
         notes         TEXT,
         created_at    TEXT    NOT NULL
       )`, []
    );

    // Lifetime aggregate counters (single row, upserted after each hand)
    await db_service.query(
      `CREATE TABLE IF NOT EXISTS BotLifetimeCounters (
         id             INTEGER PRIMARY KEY DEFAULT 1,
         total_hands    INTEGER NOT NULL DEFAULT 0,
         total_won_bb   REAL    NOT NULL DEFAULT 0,
         call_total     INTEGER NOT NULL DEFAULT 0,
         call_wins      INTEGER NOT NULL DEFAULT 0,
         bluff_total    INTEGER NOT NULL DEFAULT 0,
         bluff_success  INTEGER NOT NULL DEFAULT 0,
         three_bet_total INTEGER NOT NULL DEFAULT 0,
         three_bet_wins  INTEGER NOT NULL DEFAULT 0,
         sd_total       INTEGER NOT NULL DEFAULT 0,
         sd_wins        INTEGER NOT NULL DEFAULT 0,
         updated_at     TEXT    NOT NULL DEFAULT ''
       )`, []
    );

    // Seed the counters row if it doesn't exist yet
    await db_service.query(
      `INSERT OR IGNORE INTO BotLifetimeCounters (id, updated_at) VALUES (1, '')`, []
    );
  }

  private async loadLifetimeStats(): Promise<void> {
    const rows = await db_service.query(
      `SELECT * FROM BotLifetimeCounters WHERE id = 1`, []
    );
    if (rows.length > 0) {
      const r = rows[0] as any;
      this.lifetime = {
        totalHands:    r.total_hands    ?? 0,
        totalWonBB:    r.total_won_bb   ?? 0,
        callTotal:     r.call_total     ?? 0,
        callWins:      r.call_wins      ?? 0,
        bluffTotal:    r.bluff_total    ?? 0,
        bluffSuccess:  r.bluff_success  ?? 0,
        threeBetTotal: r.three_bet_total ?? 0,
        threeBetWins:  r.three_bet_wins  ?? 0,
        showdownTotal: r.sd_total       ?? 0,
        showdownWins:  r.sd_wins        ?? 0,
      };
    }
  }

  private async saveHandToDB(outcome: HandOutcome): Promise<void> {
    await db_service.query(
      `INSERT INTO BotHandHistory
         (session_id, hand_id, street, position, hand_key, action,
          size_bb, pot_bb, won_bb, showdown, villain_vpip, villain_pfr, notes, created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      [
        this.sessionId,
        outcome.handId,
        outcome.street,
        outcome.heroPosition,
        outcome.handKey,
        outcome.action,
        outcome.sizeBB,
        outcome.potBB,
        isNaN(outcome.wonBB) ? 0 : outcome.wonBB,   // ← guard against NaN
        outcome.showdown ? 1 : 0,
        outcome.villainVPIP ?? null,
        outcome.villainPFR  ?? null,
        outcome.notes       ?? null,
        new Date().toISOString(),
      ]
    );

    // Update lifetime counters row in-place
    await this.saveLifetimeCounters();
  }

  private async saveLifetimeCounters(): Promise<void> {
    await db_service.query(
      `UPDATE BotLifetimeCounters SET
         total_hands     = ?,
         total_won_bb    = ?,
         call_total      = ?,
         call_wins       = ?,
         bluff_total     = ?,
         bluff_success   = ?,
         three_bet_total = ?,
         three_bet_wins  = ?,
         sd_total        = ?,
         sd_wins         = ?,
         updated_at      = ?
       WHERE id = 1`,
      [
        this.lifetime.totalHands,
        this.lifetime.totalWonBB,
        this.lifetime.callTotal,
        this.lifetime.callWins,
        this.lifetime.bluffTotal,
        this.lifetime.bluffSuccess,
        this.lifetime.threeBetTotal,
        this.lifetime.threeBetWins,
        this.lifetime.showdownTotal,
        this.lifetime.showdownWins,
        new Date().toISOString(),
      ]
    );
  }

  private lifetimeBBPer100(): number {
    if (this.lifetime.totalHands === 0) return 0;
    return (this.lifetime.totalWonBB / this.lifetime.totalHands) * 100;
  }

  private updateBias(): void {
    const hands = this.lifetime.totalHands;
    const stats = this.getSessionStats();

    // Prefer lifetime data once we have enough; use session data early
    const bluffRate = this.lifetime.bluffTotal >= 10
      ? this.lifetime.bluffSuccess / this.lifetime.bluffTotal
      : this.bluffTotal > 0 ? this.bluffSuccess / this.bluffTotal : null;

    const lifeBB100 = this.lifetimeBBPer100();

    if (stats.bbPer100 < -15 || (hands >= 50 && lifeBB100 < -10)) {
      this.currentBias = "tight";
    } else if (bluffRate !== null && bluffRate < 0.36) {
      this.currentBias = "tight";
    } else if (stats.bbPer100 > 8 || (hands >= 50 && lifeBB100 > 5)) {
      this.currentBias = "gto";
    } else {
      this.currentBias = "gto";
    }
  }

}