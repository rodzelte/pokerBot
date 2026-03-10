import { BotAction, defaultCheckAction, defaultFoldAction } from "../interfaces/ai-client-interfaces.ts";
import { Game } from "../models/game.ts";
import { Table } from "../models/table.ts";
import { PlayerAction } from "../models/player-action.ts";
import { PlayerStats } from "../models/player-stats.ts";

type PreflopSpot =
    | "unopened"
    | "limped"
    | "raised"
    | "reraised"
    | "jammed";

type VillainProfile = {
    totalHands: number;
    vpip: number;
    pfr: number;
};

export class LocalPokerEngine {
    public decide(game: Game, table: Table): BotAction {
        const hero = game.getHero();
        if (!hero) return defaultFoldAction;

        const hand = hero.getHand() ?? [];
        const heroId = hero.getPlayerId();
        const heroStackBB = hero.getStackSize();

        if (!hand.length || heroStackBB === undefined) {
            return defaultFoldAction;
        }

        const street = table.getStreet();
        const heroPosition = table.getPlayerPositionFromId(heroId);
        const playerActions = table.getPlayerActions();
        const board = this.getBoard(table);
        const effectiveStackBB = this.getEffectiveStackBB(table, heroId, heroStackBB);
        const villainProfile = this.getMainVillainProfile(table, heroId);

        if (street === "preflop") {
            return this.decidePreflop(
                hand,
                heroPosition,
                table.getNumPlayers(),
                playerActions,
                effectiveStackBB,
                villainProfile
            );
        }

    return this.decidePostflop(
    hand,
    board,
    playerActions,
    heroPosition,
    effectiveStackBB,
    villainProfile,
    table.getPot()
);
    }

 private decidePreflop(
    hand: string[],
    heroPosition: string,
    numPlayers: number,
    actions: PlayerAction[],
    effectiveStackBB: number,
    villain: VillainProfile | null
): BotAction {
    const handKey = this.normalizeHand(hand);
    const spot = this.classifyPreflopSpot(actions);
    const openRange = this.getOpenRange(heroPosition, numPlayers);
    const isoRange = this.getIsoRange(heroPosition, numPlayers);
    const defendRange = this.getDefendVsRaiseRange(heroPosition, effectiveStackBB, villain);
    const jamCallRange = this.getCallVsJamRange(effectiveStackBB, villain);

    const limpers = this.countLimpers(actions);
    const lastRaiseSizeBB = this.getLastRaiseSizeBB(actions);

    if (spot === "unopened") {
        if (openRange.has(handKey)) {
            return {
                action_str: "raise",
                bet_size_in_BBs: this.getOpenSizeBB(heroPosition, numPlayers, effectiveStackBB)
            };
        }

        if (heroPosition === "BB") {
            return defaultCheckAction;
        }

        return defaultFoldAction;
    }

    if (spot === "limped") {
        if (heroPosition === "BB") {
            if (isoRange.has(handKey)) {
                return {
                    action_str: "raise",
                    bet_size_in_BBs: this.getIsoSizeBB(limpers, heroPosition, effectiveStackBB, numPlayers)
                };
            }

            return defaultCheckAction;
        }

        if (isoRange.has(handKey)) {
            return {
                action_str: "raise",
                bet_size_in_BBs: this.getIsoSizeBB(limpers, heroPosition, effectiveStackBB, numPlayers)
            };
        }

        return defaultCheckAction;
    }

    if (spot === "raised" || spot === "reraised") {
        if (defendRange.has(handKey)) {
            if (this.shouldThreeBet(handKey, effectiveStackBB, villain)) {
                return {
                    action_str: "raise",
                    bet_size_in_BBs: this.getThreeBetSizeBB(lastRaiseSizeBB || 2.5, heroPosition, effectiveStackBB)
                };
            }

            return {
                action_str: "call",
                bet_size_in_BBs: 1
            };
        }

        return defaultFoldAction;
    }

    if (spot === "jammed") {
        if (jamCallRange.has(handKey)) {
            return {
                action_str: "call",
                bet_size_in_BBs: 1
            };
        }

        return defaultFoldAction;
    }

    return defaultFoldAction;
}
   private decidePostflop(
    hand: string[],
    board: string[],
    actions: PlayerAction[],
    heroPosition: string,
    effectiveStackBB: number,
    villain: VillainProfile | null,
    potBB: number = 0
): BotAction {
    const facingBet = this.isFacingBet(actions);
    const handClass = this.classifyPostflopHand(hand, board);

    if (!facingBet) {
        if (handClass === "monster" || handClass === "strongMade" || handClass === "draw") {
            return {
                action_str: "bet",
                bet_size_in_BBs: this.getPostflopBetSizeBB(potBB, effectiveStackBB, handClass, board)
            };
        }

        return defaultCheckAction;
    }

    if (handClass === "monster" || handClass === "strongMade") {
        return {
            action_str: "call",
            bet_size_in_BBs: 1
        };
    }

    if (handClass === "draw") {
        if ((villain && villain.totalHands >= 20 && villain.vpip > 40) || effectiveStackBB <= 20) {
            return {
                action_str: "call",
                bet_size_in_BBs: 1
            };
        }
    }

    if (heroPosition === "BB" && handClass === "weakMade") {
        return {
            action_str: "call",
            bet_size_in_BBs: 1
        };
    }

    return defaultFoldAction;
}

    private classifyPreflopSpot(actions: PlayerAction[]): PreflopSpot {
        const voluntary = actions.filter(a => {
            const action = a.getAction().toLowerCase();
            return action !== "post";
        });

        const raises = voluntary.filter(a => {
            const action = a.getAction().toLowerCase();
            return action === "raise" || action === "bet";
        });

        const calls = voluntary.filter(a => a.getAction().toLowerCase() === "call");

        if (raises.length === 0 && calls.length === 0) {
            return "unopened";
        }

        if (raises.length === 0 && calls.length > 0) {
            return "limped";
        }

        const biggestRaise = Math.max(...raises.map(a => a.getBetAmount()), 0);

        if (biggestRaise >= 10) {
            return "jammed";
        }

        if (raises.length >= 2) {
            return "reraised";
        }

        return "raised";
    }

    private isFacingBet(actions: PlayerAction[]): boolean {
        return actions.some(a => {
            const action = a.getAction().toLowerCase();
            return action === "bet" || action === "raise";
        });
    }

    private getOpenRange(position: string, numPlayers: number): Set<string> {
        if (numPlayers === 2 && position === "SB") {
            return new Set([
                "AA","KK","QQ","JJ","TT","99","88","77","66","55",
                "AKs","AQs","AJs","ATs","A9s","A8s","A7s","A6s","A5s","A4s","A3s","A2s",
                "KQs","KJs","KTs","QJs","QTs","JTs","T9s","98s","87s","76s",
                "AKo","AQo","AJo","ATo","KQo","KJo","QJo","JTo"
            ]);
        }

        const ranges: Record<string, string[]> = {
            "UTG": ["AA","KK","QQ","JJ","TT","99","AKs","AQs","AJs","KQs","AKo","AQo"],
            "UTG+1": ["AA","KK","QQ","JJ","TT","99","88","AKs","AQs","AJs","KQs","AKo","AQo","AJo"],
            "MP": ["AA","KK","QQ","JJ","TT","99","88","77","AKs","AQs","AJs","ATs","KQs","QJs","JTs","AKo","AQo","AJo","KQo"],
            "LJ": ["AA","KK","QQ","JJ","TT","99","88","77","66","AKs","AQs","AJs","ATs","KQs","KJs","QJs","JTs","T9s","AKo","AQo","AJo","KQo"],
            "HJ": ["AA","KK","QQ","JJ","TT","99","88","77","66","55","AKs","AQs","AJs","ATs","A9s","KQs","KJs","QJs","JTs","T9s","98s","AKo","AQo","AJo","KQo","QJo"],
            "CO": ["AA","KK","QQ","JJ","TT","99","88","77","66","55","44","AKs","AQs","AJs","ATs","A9s","A8s","KQs","KJs","KTs","QJs","QTs","JTs","T9s","98s","87s","AKo","AQo","AJo","ATo","KQo","KJo","QJo"],
            "BU": ["AA","KK","QQ","JJ","TT","99","88","77","66","55","44","33","22","AKs","AQs","AJs","ATs","A9s","A8s","A7s","A6s","A5s","A4s","A3s","A2s","KQs","KJs","KTs","K9s","QJs","QTs","Q9s","JTs","J9s","T9s","98s","87s","76s","AKo","AQo","AJo","ATo","KQo","KJo","QJo","JTo"],
            "SB": ["AA","KK","QQ","JJ","TT","99","88","77","66","55","44","33","22","AKs","AQs","AJs","ATs","A9s","A8s","A7s","A6s","A5s","A4s","A3s","A2s","KQs","KJs","KTs","K9s","QJs","QTs","Q9s","JTs","J9s","T9s","98s","87s","76s","65s","AKo","AQo","AJo","ATo","KQo","KJo","QJo","JTo"],
            "BB": []
        };

        return new Set(ranges[position] ?? []);
    }

    private getIsoRange(position: string, numPlayers: number): Set<string> {
        const base = this.getOpenRange(position, numPlayers);
        const extra = ["KTo","QTo","97s","86s","75s","54s"];
        if (numPlayers === 2) {
            extra.push("A7o","A6o","K9o","Q9o","J9o");
        }
        return new Set([...base, ...extra]);
    }

    private getDefendVsRaiseRange(position: string, effectiveStackBB: number, villain: VillainProfile | null): Set<string> {
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
            range = new Set([...range].filter(h =>
                ["AA","KK","QQ","JJ","TT","99","AKs","AQs","AKo","AQo"].includes(h)
            ));
        }

        if (villain && villain.totalHands >= 20 && villain.vpip > 45 && villain.pfr > 25) {
            ["A9s","KTs","QTs","J9s","T8s","ATo","KJo"].forEach(h => range.add(h));
        }

        return range;
    }

    private getCallVsJamRange(effectiveStackBB: number, villain: VillainProfile | null): Set<string> {
        let range = new Set<string>(["AA","KK","QQ","JJ","TT","AKs","AQs","AKo"]);

        if (effectiveStackBB <= 15) {
            ["99","88","AQo","AJs"].forEach(h => range.add(h));
        }

        if (villain && villain.totalHands >= 20 && villain.vpip > 50 && villain.pfr > 35) {
            ["99","88","AQo","AJs","ATs"].forEach(h => range.add(h));
        }

        return range;
    }

    private shouldThreeBet(handKey: string, effectiveStackBB: number, villain: VillainProfile | null): boolean {
        if (["AA","KK","QQ","AKs","AKo"].includes(handKey)) {
            return true;
        }

        if (effectiveStackBB <= 25 && ["JJ","TT","AQs","AQo"].includes(handKey)) {
            return true;
        }

        if (villain && villain.totalHands >= 20 && villain.vpip > 45 && villain.pfr > 30) {
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
    const ranks = board.map(c => c[0].toUpperCase());
    const order = "A23456789TJQKA";
    const idxs = ranks
        .map(r => order.indexOf(r))
        .filter(i => i >= 0)
        .sort((a, b) => a - b);

    for (const card of board) {
        const suit = card[1];
        suits.set(suit, (suits.get(suit) ?? 0) + 1);
    }

    const twoTone = Array.from(suits.values()).some(v => v >= 2);

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
    handClass: "air" | "draw" | "weakMade" | "strongMade" | "monster",
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
    } else {
        fraction = 0.33;
    }

    const size = potBB * fraction;
    return Math.min(Math.max(size, 1), effectiveStackBB);
}

    private classifyPostflopHand(hand: string[], board: string[]): "air" | "draw" | "weakMade" | "strongMade" | "monster" {
        const rankCounts = this.getRankCounts([...hand, ...board]);
        const maxCount = Math.max(...Array.from(rankCounts.values()), 1);

        if (maxCount >= 4) return "monster";
        if (maxCount >= 3) return "strongMade";

        const pairRanks = Array.from(rankCounts.values()).filter(v => v >= 2).length;
        if (pairRanks >= 2) return "strongMade";
        if (pairRanks === 1) {
            const boardTop = this.getBoardTopRank(board);
            const holeRanks = hand.map(c => c[0].toUpperCase());
            if (holeRanks.includes(boardTop)) return "strongMade";
            return "weakMade";
        }

        if (this.hasFlushDraw(hand, board) || this.hasOpenEndedStraightDraw(hand, board)) {
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
            .map(s => s.trim())
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
    const voluntaryCalls = actions.filter(a => a.getAction().toLowerCase() === "call");
    const raises = actions.filter(a => {
        const action = a.getAction().toLowerCase();
        return action === "raise" || action === "bet";
    });

    if (raises.length > 0) {
        return 0;
    }

    return voluntaryCalls.length;
}

private getLastRaiseSizeBB(actions: PlayerAction[]): number {
    const raises = actions.filter(a => {
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
            pfr
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
        const boardRanks = board.map(c => c[0].toUpperCase());
        boardRanks.sort((a, b) => order.indexOf(a) - order.indexOf(b));
        return boardRanks[0] ?? "";
    }

    private hasFlushDraw(hand: string[], board: string[]): boolean {
        const suits = new Map<string, number>();
        for (const card of [...hand, ...board]) {
            const suit = card[1];
            suits.set(suit, (suits.get(suit) ?? 0) + 1);
        }
        return Array.from(suits.values()).some(v => v === 4);
    }

    private hasOpenEndedStraightDraw(hand: string[], board: string[]): boolean {
        const order = "A23456789TJQKA";
        const uniqueRanks = Array.from(new Set([...hand, ...board].map(c => c[0].toUpperCase())));
        const idxs = uniqueRanks
            .map(r => order.indexOf(r))
            .filter(i => i >= 0)
            .sort((a, b) => a - b);

        for (let i = 0; i <= idxs.length - 4; i++) {
            const window = idxs.slice(i, i + 4);
            if (window[3] - window[0] === 3) {
                return true;
            }
        }
        return false;
    }
}