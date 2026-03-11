/**
 * villain-memory.ts
 *
 * Parses the PokerNow in-game log container and builds persistent villain
 * profiles that survive across sessions. These profiles feed directly into
 * the engine's exploitative decision-making.
 *
 * What we track per villain:
 *   - VPIP  (voluntarily put chips in pot)
 *   - PFR   (preflop raise frequency)
 *   - AF    (aggression factor: bets+raises / calls)
 *   - WTSD  (went to showdown %)
 *   - WSD   (won at showdown %)
 *   - CBet  (flop continuation bet %)
 *   - FoldToCBet
 *   - 3Bet %
 *   - FoldTo3Bet %
 *   - Limp %
 *   - totalHands
 *
 * How bluff detection works:
 *   A "BluffScore" 0–100 is computed from these stats:
 *     High AF + Low WSD  → pure bluffer
 *     High VPIP + Low PFR → passive fish (calls too much, rarely bluffs)
 *     High 3Bet + High FoldToRaise → light 3-bettor
 *     Low WTSD + High CBet → c-bet and give up
 *   The engine uses BluffScore to:
 *     1. Lower call threshold vs. high bluff scores
 *     2. Tighten up vs. low bluff scores (value-heavy villains)
 *     3. Adjust 3-bet frequencies
 */

import Database from "better-sqlite3";

// ─── Types ────────────────────────────────────────────────────────────────────

export type VillainStats = {
  playerId:       string;   // e.g. "bmoANWYlvv"
  playerName:     string;   // e.g. "BOTAI"
  totalHands:     number;
  vpipHands:      number;   // hands where villain put chips in voluntarily
  pfrHands:       number;   // hands where villain raised preflop
  threeBetHands:  number;
  foldTo3Bet:     number;
  cBetFlop:       number;   // times villain c-bet the flop after raising PF
  foldToCBet:     number;   // times villain folded to a c-bet
  totalBets:      number;   // postflop bets + raises
  totalCalls:     number;   // postflop calls
  totalFolds:     number;
  wentToShowdown: number;
  wonAtShowdown:  number;
  limpHands:      number;
  lastSeen:       number;   // unix timestamp
};

export type VillainProfile = VillainStats & {
  // Computed on read
  vpip:           number;   // %
  pfr:            number;   // %
  af:             number;   // aggression factor
  wtsd:           number;   // %
  wsd:            number;   // %
  threeBetPct:    number;   // %
  foldTo3BetPct:  number;   // %
  foldToCBetPct:  number;   // %
  limpPct:        number;   // %
  bluffScore:     number;   // 0–100 composite bluff likelihood
  playerType:     PlayerType;
};

export type PlayerType =
  | "unknown"
  | "nit"           // very tight, rarely bluffs
  | "tag"           // tight-aggressive, value-heavy
  | "lag"           // loose-aggressive, bluffs often
  | "fish"          // loose-passive, calls too much
  | "maniac"        // hyper-aggressive, over-bluffs
  | "calling-station"; // calls everything, never bluffs

// ─── Log Parser ───────────────────────────────────────────────────────────────

type HandState = {
  handId:        string;
  players:       Map<string, { id: string; name: string; stack: number }>;
  preflopRaiser: string | null;  // player who last raised preflop
  flopBettor:    string | null;  // player who bet the flop
  actions:       { player: string; action: string; amount: number; street: string }[];
  boardStreets:  string[];       // ["preflop", "flop", "turn", "river"] seen so far
  showdownPlayers: string[];
  winners:       string[];
  currentStreet: string;
};

export class LogParser {
  private readonly playerPattern  = /"([^"]+) @ ([a-zA-Z0-9]+)"/g;
  private readonly handStartPat   = /-- starting hand #(\d+) \(id: ([a-z0-9]+)\)/;
  private readonly handEndPat     = /-- ending hand #\d+/;
  private readonly stacksPat      = /Player stacks: (.+)/;
  private readonly actionPat      = /"([^"]+) @ ([a-zA-Z0-9]+)" (raises|calls|bets|checks|folds|posts)(?:\s+(?:a big blind|a small blind|an ante))?(?: to| of)?\s*([\d.]+)?/;
  private readonly streetPat      = /\*\*\* (FLOP|TURN|RIVER) \*\*\*/i;
  private readonly showdownPat    = /"([^"]+) @ ([a-zA-Z0-9]+)" shows a/;
  private readonly winnerPat      = /"([^"]+) @ ([a-zA-Z0-9]+)" collected ([\d.]+) from/;
  private readonly uncalledPat    = /Uncalled bet.*returned to "([^"]+) @ ([a-zA-Z0-9]+)"/;

  /**
   * Parse the text content of the PokerNow log container.
   * Returns a delta map: playerId → stat increments for this batch of text.
   */
  public parseLogText(logText: string): Map<string, Partial<VillainStats>> {
    const lines = logText.split("\n").map((l) => l.trim()).filter(Boolean);
    const deltas = new Map<string, Partial<VillainStats>>();

    let hand: HandState | null = null;

    const getOrCreate = (id: string, name: string): Partial<VillainStats> => {
      if (!deltas.has(id)) {
        deltas.set(id, {
          playerId: id, playerName: name,
          totalHands: 0, vpipHands: 0, pfrHands: 0, threeBetHands: 0,
          foldTo3Bet: 0, cBetFlop: 0, foldToCBet: 0,
          totalBets: 0, totalCalls: 0, totalFolds: 0,
          wentToShowdown: 0, wonAtShowdown: 0, limpHands: 0,
          lastSeen: Date.now(),
        });
      }
      return deltas.get(id)!;
    };

    // Process lines in reverse (log shows newest first)
    const reversed = [...lines].reverse();

    for (const line of reversed) {
      // ── Hand start ───────────────────────────────────────────────────────
      const startMatch = line.match(this.handStartPat);
      if (startMatch) {
        hand = {
          handId: startMatch[2],
          players: new Map(),
          preflopRaiser: null,
          flopBettor: null,
          actions: [],
          boardStreets: ["preflop"],
          showdownPlayers: [],
          winners: [],
          currentStreet: "preflop",
        };
        continue;
      }

      if (!hand) continue;

      // ── Hand end — commit stats ───────────────────────────────────────────
      if (this.handEndPat.test(line)) {
        this.commitHandStats(hand, deltas, getOrCreate);
        hand = null;
        continue;
      }

      // ── Player stacks (identifies participants) ───────────────────────────
      const stacksMatch = line.match(this.stacksPat);
      if (stacksMatch) {
        const entries = stacksMatch[1].split("|");
        for (const entry of entries) {
          const m = entry.match(/"([^"]+) @ ([a-zA-Z0-9]+)"\s*\(([\d.]+)\)/);
          if (m) {
            hand.players.set(m[2], { id: m[2], name: m[1], stack: parseFloat(m[3]) });
          }
        }
        continue;
      }

      // ── Street markers ────────────────────────────────────────────────────
      const streetMatch = line.match(this.streetPat);
      if (streetMatch) {
        hand.currentStreet = streetMatch[1].toLowerCase();
        hand.boardStreets.push(hand.currentStreet);
        continue;
      }

      // ── Showdown ──────────────────────────────────────────────────────────
      const showMatch = line.match(this.showdownPat);
      if (showMatch) {
        hand.showdownPlayers.push(showMatch[2]);
        continue;
      }

      // ── Winner ────────────────────────────────────────────────────────────
      const winMatch = line.match(this.winnerPat);
      if (winMatch) {
        hand.winners.push(winMatch[2]);
        continue;
      }

      // ── Actions ───────────────────────────────────────────────────────────
      const actMatch = line.match(this.actionPat);
      if (actMatch) {
        const [, name, id, action, amountStr] = actMatch;
        const amount = parseFloat(amountStr ?? "0") || 0;
        hand.actions.push({
          player: id,
          action: action.toLowerCase(),
          amount,
          street: hand.currentStreet,
        });

        // Track preflop raiser for c-bet detection
        if (hand.currentStreet === "preflop" && action === "raises") {
          hand.preflopRaiser = id;
        }
        // Track flop bettor for fold-to-cbet
        if (hand.currentStreet === "flop" && action === "bets" && !hand.flopBettor) {
          hand.flopBettor = id;
        }
      }
    }

    return deltas;
  }

  private commitHandStats(
    hand: HandState,
    deltas: Map<string, Partial<VillainStats>>,
    getOrCreate: (id: string, name: string) => Partial<VillainStats>
  ): void {
    const preflopActions = hand.actions.filter((a) => a.street === "preflop");
    const postflopActions = hand.actions.filter((a) => a.street !== "preflop");

    // Identify who raised preflop (for 3-bet detection)
    const preflopRaises = preflopActions.filter((a) => a.action === "raises");

    // Count hands for all participants
    for (const [id, p] of hand.players) {
      const d = getOrCreate(id, p.name);
      d.totalHands = (d.totalHands ?? 0) + 1;
    }

    // VPIP: player put chips in voluntarily preflop (call or raise, not post)
    const vpipSet = new Set<string>();
    for (const a of preflopActions) {
      if (a.action === "calls" || a.action === "raises") {
        vpipSet.add(a.player);
      }
    }
    for (const id of vpipSet) {
      const p = hand.players.get(id);
      if (!p) continue;
      const d = getOrCreate(id, p.name);
      d.vpipHands = (d.vpipHands ?? 0) + 1;
    }

    // PFR: raised preflop
    const pfrSet = new Set(preflopRaises.map((a) => a.player));
    for (const id of pfrSet) {
      const p = hand.players.get(id);
      if (!p) continue;
      const d = getOrCreate(id, p.name);
      d.pfrHands = (d.pfrHands ?? 0) + 1;
    }

    // Limp: called preflop without raising before
    for (const a of preflopActions) {
      if (a.action === "calls" && !pfrSet.has(a.player)) {
        const p = hand.players.get(a.player);
        if (p) getOrCreate(a.player, p.name).limpHands =
          (getOrCreate(a.player, p.name).limpHands ?? 0) + 1;
      }
    }

    // 3-bet: second raise preflop
    if (preflopRaises.length >= 2) {
      const threeBettor = preflopRaises[1].player;
      const p = hand.players.get(threeBettor);
      if (p) {
        const d = getOrCreate(threeBettor, p.name);
        d.threeBetHands = (d.threeBetHands ?? 0) + 1;
      }
      // Fold to 3-bet: original raiser folded after the 3-bet
      const origRaiser = preflopRaises[0].player;
      const foldAfter3Bet = preflopActions.find(
        (a) => a.action === "folds" && a.player === origRaiser
      );
      if (foldAfter3Bet) {
        const p2 = hand.players.get(origRaiser);
        if (p2) {
          const d = getOrCreate(origRaiser, p2.name);
          d.foldTo3Bet = (d.foldTo3Bet ?? 0) + 1;
        }
      }
    }

    // C-bet flop: preflop raiser bets the flop
    if (hand.preflopRaiser && hand.flopBettor === hand.preflopRaiser) {
      const p = hand.players.get(hand.preflopRaiser);
      if (p) {
        const d = getOrCreate(hand.preflopRaiser, p.name);
        d.cBetFlop = (d.cBetFlop ?? 0) + 1;
      }
    }

    // Fold to c-bet: someone folded on the flop when facing a bet
    if (hand.flopBettor) {
      const flopFolds = hand.actions.filter(
        (a) => a.street === "flop" && a.action === "folds" && a.player !== hand.flopBettor
      );
      for (const f of flopFolds) {
        const p = hand.players.get(f.player);
        if (p) {
          const d = getOrCreate(f.player, p.name);
          d.foldToCBet = (d.foldToCBet ?? 0) + 1;
        }
      }
    }

    // Postflop aggression
    for (const a of postflopActions) {
      const p = hand.players.get(a.player);
      if (!p) continue;
      const d = getOrCreate(a.player, p.name);
      if (a.action === "bets" || a.action === "raises") {
        d.totalBets = (d.totalBets ?? 0) + 1;
      } else if (a.action === "calls") {
        d.totalCalls = (d.totalCalls ?? 0) + 1;
      } else if (a.action === "folds") {
        d.totalFolds = (d.totalFolds ?? 0) + 1;
      }
    }

    // Showdown / WSD
    for (const id of hand.showdownPlayers) {
      const p = hand.players.get(id);
      if (!p) continue;
      const d = getOrCreate(id, p.name);
      d.wentToShowdown = (d.wentToShowdown ?? 0) + 1;
      if (hand.winners.includes(id)) {
        d.wonAtShowdown = (d.wonAtShowdown ?? 0) + 1;
      }
    }
  }
}

// ─── Villain Memory (SQLite persistence) ─────────────────────────────────────

export class VillainMemory {
  private db: Database.Database;
  private readonly parser = new LogParser();

  constructor(dbPath = "./app/pokernow-gpt.db") {
    this.db = new Database(dbPath);
    this.createTable();
  }

  private createTable(): void {
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS VillainMemory (
        player_id        TEXT    PRIMARY KEY,
        player_name      TEXT    NOT NULL,
        total_hands      INTEGER NOT NULL DEFAULT 0,
        vpip_hands       INTEGER NOT NULL DEFAULT 0,
        pfr_hands        INTEGER NOT NULL DEFAULT 0,
        three_bet_hands  INTEGER NOT NULL DEFAULT 0,
        fold_to_3bet     INTEGER NOT NULL DEFAULT 0,
        cbet_flop        INTEGER NOT NULL DEFAULT 0,
        fold_to_cbet     INTEGER NOT NULL DEFAULT 0,
        total_bets       INTEGER NOT NULL DEFAULT 0,
        total_calls      INTEGER NOT NULL DEFAULT 0,
        total_folds      INTEGER NOT NULL DEFAULT 0,
        went_to_showdown INTEGER NOT NULL DEFAULT 0,
        won_at_showdown  INTEGER NOT NULL DEFAULT 0,
        limp_hands       INTEGER NOT NULL DEFAULT 0,
        last_seen        INTEGER NOT NULL DEFAULT 0
      )
    `);
  }

  /**
   * Feed raw log text (from the PokerNow log container) into the memory.
   * Call this periodically from bot.ts — e.g. every 5 hands.
   */
  public ingestLog(logText: string): void {
    const deltas = this.parser.parseLogText(logText);
    const upsert = this.db.prepare(`
      INSERT INTO VillainMemory (
        player_id, player_name, total_hands, vpip_hands, pfr_hands,
        three_bet_hands, fold_to_3bet, cbet_flop, fold_to_cbet,
        total_bets, total_calls, total_folds,
        went_to_showdown, won_at_showdown, limp_hands, last_seen
      ) VALUES (
        @player_id, @player_name, @total_hands, @vpip_hands, @pfr_hands,
        @three_bet_hands, @fold_to_3bet, @cbet_flop, @fold_to_cbet,
        @total_bets, @total_calls, @total_folds,
        @went_to_showdown, @won_at_showdown, @limp_hands, @last_seen
      )
      ON CONFLICT(player_id) DO UPDATE SET
        player_name      = excluded.player_name,
        total_hands      = VillainMemory.total_hands      + excluded.total_hands,
        vpip_hands       = VillainMemory.vpip_hands       + excluded.vpip_hands,
        pfr_hands        = VillainMemory.pfr_hands        + excluded.pfr_hands,
        three_bet_hands  = VillainMemory.three_bet_hands  + excluded.three_bet_hands,
        fold_to_3bet     = VillainMemory.fold_to_3bet     + excluded.fold_to_3bet,
        cbet_flop        = VillainMemory.cbet_flop        + excluded.cbet_flop,
        fold_to_cbet     = VillainMemory.fold_to_cbet     + excluded.fold_to_cbet,
        total_bets       = VillainMemory.total_bets       + excluded.total_bets,
        total_calls      = VillainMemory.total_calls      + excluded.total_calls,
        total_folds      = VillainMemory.total_folds      + excluded.total_folds,
        went_to_showdown = VillainMemory.went_to_showdown + excluded.went_to_showdown,
        won_at_showdown  = VillainMemory.won_at_showdown  + excluded.won_at_showdown,
        limp_hands       = VillainMemory.limp_hands       + excluded.limp_hands,
        last_seen        = excluded.last_seen
    `);

    const tx = this.db.transaction((entries: [string, Partial<VillainStats>][]) => {
      for (const [, d] of entries) {
        if (!d.playerId || !d.playerName) continue;
        upsert.run({
          player_id:        d.playerId,
          player_name:      d.playerName,
          total_hands:      d.totalHands      ?? 0,
          vpip_hands:       d.vpipHands       ?? 0,
          pfr_hands:        d.pfrHands        ?? 0,
          three_bet_hands:  d.threeBetHands   ?? 0,
          fold_to_3bet:     d.foldTo3Bet      ?? 0,
          cbet_flop:        d.cBetFlop        ?? 0,
          fold_to_cbet:     d.foldToCBet      ?? 0,
          total_bets:       d.totalBets       ?? 0,
          total_calls:      d.totalCalls      ?? 0,
          total_folds:      d.totalFolds      ?? 0,
          went_to_showdown: d.wentToShowdown  ?? 0,
          won_at_showdown:  d.wonAtShowdown   ?? 0,
          limp_hands:       d.limpHands       ?? 0,
          last_seen:        d.lastSeen        ?? Date.now(),
        });
      }
    });

    tx([...deltas.entries()]);
    console.log(`[VillainMemory] Ingested ${deltas.size} players from log.`);
  }

  /** Get full computed profile for a player by ID. */
  public getProfile(playerId: string): VillainProfile | null {
    const row = this.db.prepare(
      "SELECT * FROM VillainMemory WHERE player_id = ?"
    ).get(playerId) as any;
    if (!row) return null;
    return this.computeProfile(row);
  }

  /** Get profile by display name (fuzzy — use ID when possible). */
  public getProfileByName(name: string): VillainProfile | null {
    const row = this.db.prepare(
      "SELECT * FROM VillainMemory WHERE player_name = ? ORDER BY total_hands DESC LIMIT 1"
    ).get(name) as any;
    if (!row) return null;
    return this.computeProfile(row);
  }

  /** All known profiles sorted by hands played desc. */
  public getAllProfiles(): VillainProfile[] {
    const rows = this.db.prepare(
      "SELECT * FROM VillainMemory ORDER BY total_hands DESC"
    ).all() as any[];
    return rows.map((r) => this.computeProfile(r));
  }

  /** Print a quick HUD-style summary to console. */
  public printHUD(playerId: string): void {
    const p = this.getProfile(playerId);
    if (!p) { console.log(`[HUD] No data for ${playerId}`); return; }
    console.log(
      `[HUD] ${p.playerName} (${p.totalHands} hands) | ` +
      `VPIP:${p.vpip.toFixed(0)}% PFR:${p.pfr.toFixed(0)}% ` +
      `AF:${p.af.toFixed(1)} 3B:${p.threeBetPct.toFixed(0)}% ` +
      `WTSD:${p.wtsd.toFixed(0)}% WSD:${p.wsd.toFixed(0)}% ` +
      `FCB:${p.foldToCBetPct.toFixed(0)}% ` +
      `BluffScore:${p.bluffScore.toFixed(0)} [${p.playerType}]`
    );
  }

  // ── Profile computation ───────────────────────────────────────────────────

  private computeProfile(row: any): VillainProfile {
    const h  = row.total_hands   || 1; // avoid div/0
    const pf = row.vpip_hands    || 0;

    const vpip        = (row.vpip_hands       / h)  * 100;
    const pfr         = (row.pfr_hands        / h)  * 100;
    const threeBetPct = (row.three_bet_hands  / Math.max(row.pfr_hands, 1)) * 100;
    const foldTo3BetPct = (row.fold_to_3bet   / Math.max(row.pfr_hands, 1)) * 100;
    const foldToCBetPct = (row.fold_to_cbet   / Math.max(row.cbet_flop || h * 0.3, 1)) * 100;
    const limpPct     = (row.limp_hands       / h)  * 100;
    const wtsd        = (row.went_to_showdown / h)  * 100;
    const wsd         = row.went_to_showdown > 0
      ? (row.won_at_showdown / row.went_to_showdown) * 100
      : 50;

    // AF = (bets + raises) / calls  — infinity if no calls (hyper-aggressive)
    const af = row.total_calls > 0
      ? (row.total_bets) / row.total_calls
      : row.total_bets > 0 ? 10 : 1;

    // ── BluffScore 0–100 ──────────────────────────────────────────────────
    // High score = likely bluffing, low score = value-heavy
    let bluffScore = 50; // start neutral

    // High AF with low WSD → bets a lot but folds at showdown (bluff-heavy)
    if (af > 3 && wtsd < 25)  bluffScore += 20;
    else if (af > 2)           bluffScore += 10;
    else if (af < 1)           bluffScore -= 15;

    // High VPIP + High PFR → aggressive (can be bluffing)
    if (vpip > 40 && pfr > 25) bluffScore += 15;
    // High VPIP + Low PFR  → passive fish (calling station, NOT bluffing)
    else if (vpip > 40 && pfr < 15) bluffScore -= 20;

    // Low WSD → folds rivers (bluffs and gives up often)
    if (wtsd < 20) bluffScore += 10;
    else if (wtsd > 40) bluffScore -= 10; // goes to showdown → has real hands

    // High 3-bet + high fold-to-3-bet target → light 3-bettor
    if (threeBetPct > 12) bluffScore += 10;

    // Folds to c-bets a lot → not a calling station, may bluff-bet and fold
    if (foldToCBetPct > 60) bluffScore += 8;
    else if (foldToCBetPct < 30) bluffScore -= 8;

    // Low hands = unreliable data → pull toward 50
    if (h < 20) bluffScore = bluffScore * 0.5 + 50 * 0.5;

    bluffScore = Math.max(0, Math.min(100, bluffScore));

    // ── Player type classification ────────────────────────────────────────
    const playerType = this.classifyPlayerType(vpip, pfr, af, wtsd, h);

    const stats: VillainStats = {
      playerId:       row.player_id,
      playerName:     row.player_name,
      totalHands:     row.total_hands,
      vpipHands:      row.vpip_hands,
      pfrHands:       row.pfr_hands,
      threeBetHands:  row.three_bet_hands,
      foldTo3Bet:     row.fold_to_3bet,
      cBetFlop:       row.cbet_flop,
      foldToCBet:     row.fold_to_cbet,
      totalBets:      row.total_bets,
      totalCalls:     row.total_calls,
      totalFolds:     row.total_folds,
      wentToShowdown: row.went_to_showdown,
      wonAtShowdown:  row.won_at_showdown,
      limpHands:      row.limp_hands,
      lastSeen:       row.last_seen,
    };

    return {
      ...stats,
      vpip, pfr, af, wtsd, wsd,
      threeBetPct, foldTo3BetPct, foldToCBetPct, limpPct,
      bluffScore, playerType,
    };
  }

  private classifyPlayerType(
    vpip: number, pfr: number, af: number, wtsd: number, hands: number
  ): PlayerType {
    if (hands < 15) return "unknown";
    if (vpip < 15 && pfr < 10)              return "nit";
    if (vpip < 25 && pfr > 15 && af > 2)    return "tag";
    if (vpip > 35 && pfr > 20 && af > 2.5)  return "lag";
    if (vpip > 40 && pfr < 15)              return "fish";
    if (vpip > 50 && af > 4)                return "maniac";
    if (wtsd > 40 && af < 1.5)             return "calling-station";
    return "tag"; // default
  }

  public close(): void { this.db.close(); }
}

// ─── Engine Integration Helpers ───────────────────────────────────────────────

/**
 * Convert a VillainProfile into the call/bluff threshold adjustments
 * that the engine needs. Call this in buildDecisionContext.
 */
export function getEngineAdjustments(profile: VillainProfile | null): {
  callThresholdDelta: number;   // add to profile.callEquityThreshold (negative = call more)
  bluffFreqDelta:     number;   // add to flopCBetAir etc (positive = bluff more)
  threeBetFreqDelta:  number;   // add to lightThreeBet
  foldToCBet:         boolean;  // true = villain folds to c-bets, bluff more
} {
  if (!profile || profile.totalHands < 15) {
    return { callThresholdDelta: 0, bluffFreqDelta: 0, threeBetFreqDelta: 0, foldToCBet: false };
  }

  const bluff = profile.bluffScore;

  // vs. high bluff score: call more (lower threshold), bluff back less
  // vs. low bluff score: fold more (raise threshold), bluff less
  const callThresholdDelta =
    bluff > 70 ? -0.12 :   // call much more vs confirmed bluffer
    bluff > 55 ? -0.06 :   // call a bit more
    bluff < 30 ? +0.08 :   // fold more vs value-heavy
    bluff < 40 ? +0.04 :
    0;

  // Bluff more vs players who fold to c-bets
  const bluffFreqDelta =
    profile.foldToCBetPct > 65 ?  0.20 :
    profile.foldToCBetPct > 50 ?  0.10 :
    profile.foldToCBetPct < 25 ? -0.15 :
    0;

  // 3-bet more vs loose openers; less vs tight players
  const threeBetFreqDelta =
    profile.vpip > 40 && profile.pfr > 20 ?  0.10 :
    profile.vpip < 20                      ? -0.08 :
    0;

  const foldToCBet = profile.foldToCBetPct > 55;

  return { callThresholdDelta, bluffFreqDelta, threeBetFreqDelta, foldToCBet };
}