import prompt from 'prompt-sync';
import { HandOutcome } from './services/session-tracker.ts';

import { sleep } from './helpers/bot-helper.ts';
import { LocalPokerEngine } from "./services/local-poker-engine.ts";

import { AIMessage, AIService, BotAction, defaultCheckAction, defaultFoldAction } from './interfaces/ai-client-interfaces.ts';
import { ProcessedLogs } from './interfaces/log-processing-interfaces.ts';
import { constructQuery } from './helpers/construct-query-helper.ts';
import type {
    GameStateSnapshot,
    PlayerInfo,
    ActionRecord,
    PokerStreet,
    AlgorithmType,
    EnsembleConfig,
} from './interfaces/game-state-interfaces.ts';
import type { AlgorithmConfig } from './interfaces/config-interfaces.ts';

import { Game } from './models/game.ts';
import { Table } from './models/table.ts';

import { LogService } from './services/log-service.ts';
import { PlayerService } from './services/player-service.ts';
import { PuppeteerService } from './services/puppeteer-service.ts';
import { VillainMemory, getEngineAdjustments } from './services/villain-memory.ts';

import { EquityEstimator } from './services/equity-estimator.ts';
import { HandEvaluator } from './services/hand-evaluator.ts';
import { selectProfile } from './services/gto-strategy-profiles.ts';
import { RLCardService, type RLCardFullDecision } from './services/ai/rlcard-service.ts';

import { DebugMode, logResponse } from './utils/error-handling-utils.ts';
import { postProcessLogs, postProcessLogsAfterHand, preProcessLogs } from './utils/log-processing-utils.ts';
import {
    getIdToInitialStackFromMsg,
    getIdToNameFromMsg,
    getIdToTableSeatFromMsg,
    getNameToIdFromMsg,
    getPlayerStacksMsg,
    getTableSeatToIdFromMsg,
    validateAllMsg
} from './utils/message-processing-utils.ts';
import { convertToBBs, convertToValue } from './utils/value-conversion-utils.ts';

// Street lookup: number of board cards → street name
const BOARD_LENGTH_TO_STREET: Record<number, string> = {
    0: "preflop",
    3: "flop",
    4: "turn",
    5: "river",
};

export class Bot {
    private log_service: LogService;
    private player_service: PlayerService;
    private puppeteer_service: PuppeteerService;

    private game_id: string;
    private debug_mode: DebugMode;
    private local_poker_engine: LocalPokerEngine;
    private readonly villainMemory = new VillainMemory("./app/pokernow-gpt.db");
    private handsSinceLogScrape = 0;
    private readonly LOG_SCRAPE_INTERVAL = 5;
    private readonly aiService: AIService | null;
    private readonly rlcardService: RLCardService;
    private rlcardAvailable = false;
    private aiMessages: AIMessage[] = [];
    private readonly equityEstimator = new EquityEstimator();
    private readonly handEvaluator = new HandEvaluator();

    /** Algorithm selection/ensemble configuration */
    private readonly algorithmConfig: AlgorithmConfig | null;
    /** Hand counter for unique hand IDs */
    private handCounter = 0;

    private first_created: string;

    private table!: Table;
    private game!: Game;
    private bot_name!: string;

    constructor(
        log_service: LogService,
        player_service: PlayerService,
        puppeteer_service: PuppeteerService,
        game_id: string,
        debug_mode: DebugMode,
        ai_service: AIService | null = null,
        algorithmConfig: AlgorithmConfig | null = null
    ) {
        this.log_service = log_service;
        this.player_service = player_service;
        this.puppeteer_service = puppeteer_service;
        this.local_poker_engine = new LocalPokerEngine();

        this.game_id = game_id;
        this.debug_mode = debug_mode;
        this.aiService = ai_service;
        this.algorithmConfig = algorithmConfig;
        this.rlcardService = new RLCardService();
        this.rlcardService.init();

        this.first_created = "";
    }

    public async run() {
        await this.openGame();
        await this.local_poker_engine.sessionTracker.init();
        // Check if rlcard Python server is available
        this.rlcardAvailable = await this.rlcardService.isHealthy();
        if (this.rlcardAvailable) {
            console.log("[RLCard] Python server connected — rlcard advisor enabled.");
            const agentType = this.rlcardService.getAgentType();
            const available = this.rlcardService.getAvailableAgents();
            console.log(`[RLCard] Active agent: ${agentType}`);
            if (available.length > 0) {
                console.log(`[RLCard] Available agents: ${available.join(", ")}`);
            }
            // Log algorithm configuration
            if (this.algorithmConfig) {
                console.log(`[Algorithm] Mode: ${this.algorithmConfig.selection_mode}`);
                console.log(`[Algorithm] Primary: ${this.algorithmConfig.primary}`);
                console.log(`[Algorithm] Algorithms: ${this.algorithmConfig.algorithms.join(", ")}`);
                if (this.algorithmConfig.selection_mode === "ensemble" && this.algorithmConfig.ensemble_weights) {
                    console.log(`[Algorithm] Ensemble weights: ${this.algorithmConfig.ensemble_weights.join(", ")}`);
                }
                if (this.algorithmConfig.selection_mode === "cascade" && this.algorithmConfig.cascade_min_confidence) {
                    console.log(`[Algorithm] Cascade min confidence: ${(this.algorithmConfig.cascade_min_confidence * 100).toFixed(0)}%`);
                }
            }
        } else {
            console.log("[RLCard] Python server not available — running without rlcard.");
            console.log("[RLCard] To enable: cd python && pip install -r requirements.txt && python rlcard_server.py");
        }
        await this.enterTableInProgress();
        await scrapeLog(this.puppeteer_service, this.villainMemory);

        process.on("SIGINT", async () => {
            await this.local_poker_engine.sessionTracker.finaliseSession();
            this.villainMemory.close();
            this.local_poker_engine.printSessionSummary();
            process.exit(0);
        });

        await this.updateNumPlayers();

        while (true) {
            await this.waitForNextHand();
            this.aiMessages = []; // Reset AI conversation each hand
            await this.updateNumPlayers();
            await this.updateGameInfo();
            await this.updatePositionsFromDealer();
            console.log("Number of players in game:", this.table.getNumPlayers());
            this.table.setPlayersInPot(this.table.getNumPlayers());
            await this.playOneHand();
            this.handsSinceLogScrape++;
            if (this.handsSinceLogScrape >= this.LOG_SCRAPE_INTERVAL) {
                this.handsSinceLogScrape = 0;
                await scrapeLog(this.puppeteer_service, this.villainMemory);
            }
            this.table.nextHand();
            this.local_poker_engine.printSessionSummary();
        }
    }

    private async openGame() {
        console.log(`The PokerNow game with id: ${this.game_id} will now open.`);
        
        logResponse(await this.puppeteer_service.navigateToGame(this.game_id), this.debug_mode);
        logResponse(await this.puppeteer_service.waitForGameInfo(), this.debug_mode);
    
        console.log("Getting game info.");
        const res = await this.puppeteer_service.getGameInfo();
        logResponse(res, this.debug_mode);
        if (res.code == "success") {
            const game_info = this.puppeteer_service.convertGameInfo(res.data as string);
            console.log(`[Game] Blinds detected — SB: ${game_info.small_blind} / BB: ${game_info.big_blind} (raw chip values)`);
            this.table = new Table(this.player_service);
            this.game = new Game(
                this.game_id, this.table,
                game_info.big_blind, game_info.small_blind,
                game_info.game_type, 30
            );
        } else {
            throw new Error("Failed to get game info.");
        }
    }

    private async enterTableInProgress() {
        const io = prompt();
        while (true) {
            const name = io("What is your desired player name? ");
            console.log(`Your player name will be ${name}.`);
            this.bot_name = name;

            const stack_size = io("What is your desired stack size? ");
            console.log(`Your initial stack size will be ${stack_size}.`);

            console.log(`Attempting to enter table with name: ${name} and stack size: ${stack_size}.`);
            const code = logResponse(
                await this.puppeteer_service.sendEnterTableRequest(name, Number(stack_size)),
                this.debug_mode
            );

            const startingStackBB = convertToBBs(Number(stack_size), this.game.getBigBlind());
            console.log(`[Game] Starting stack: ${stack_size} chips = ${startingStackBB.toFixed(2)} BB (BB=${this.game.getBigBlind()})`);
            this.local_poker_engine.setStartingStack(startingStackBB);

            if (code === "success") break;
            console.log("Please try again.");
        }
        console.log("Waiting for table host to accept ingress request.");
        logResponse(await this.puppeteer_service.waitForTableEntry(), this.debug_mode);
    }

    private async updateNumPlayers() {
        const res = await this.puppeteer_service.getNumPlayers();
        if (res.code === "success") {
            this.table.setNumPlayers(Number(res.data));
        }
    }

    /**
     * Detect the dealer button from the DOM and assign proper poker positions
     * (SB, BB, UTG, MP, CO, BU, etc.) to all seated players.
     * Falls back to the existing log-based method if dealer is undetectable.
     */
    private async updatePositionsFromDealer() {
        const dealerRes = await this.puppeteer_service.getDealerSeat();
        if (dealerRes.code === "success") {
            const dealerSeat = dealerRes.data as number;
            this.table.setPositionsFromDealer(dealerSeat);
        } else {
            console.warn("[Bot] Could not detect dealer from DOM — falling back to log-based positions.");
        }
    }

    private async waitForNextHand() {
        console.log("Waiting for next hand to start.");
        await this.puppeteer_service.waitForNextHand(
            this.table.getNumPlayers(),
            this.game.getMaxTurnLength()
        );
    }

    private async playOneHand() {
        let processed_logs = {
            valid_msgs: new Array<Array<string>>(),
            last_created: this.first_created,
            first_fetch: true
        };

        while (true) {
            console.log("Checking for bot's turn or winner of hand.");

            const res = await this.puppeteer_service.waitForBotTurnOrWinner(
                this.table.getNumPlayers(),
                this.game.getMaxTurnLength()
            );

            if (res.code == "success") {
                const data = res.data as string;

                // ── BOT'S TURN ─────────────────────────────────────────────
                if (data.includes("action-signal")) {
                    try {
                        await sleep(2000);
                        processed_logs = await this.pullAndProcessLogs(
                            processed_logs.last_created,
                            processed_logs.first_fetch
                        );
                    } catch (err) {
                        console.log("Failed to pull logs.");
                    }

                    console.log("Performing bot's turn.");

                    // Gather table state
                    const pot_size   = await this.getPotSize();
                    const hand       = await this.getHand();
                    const stack_size = await this.getStackSize();

                    const potBB      = convertToBBs(pot_size, this.game.getBigBlind());
                    const stackBB    = convertToBBs(stack_size, this.game.getBigBlind());

                    console.log("Hero Hand:", hand.join(" "));
                    console.log(`[Game] Pot: ${pot_size} chips = ${potBB.toFixed(2)} BB | Stack: ${stack_size} chips = ${stackBB.toFixed(2)} BB | BB size: ${this.game.getBigBlind()}`);

                    this.table.setPot(potBB);
                    await this.updateHero(hand, stackBB);

                    await this.updateBoardState();

                    // ── Hybrid equity estimation (pokersolver / poker-odds-calc / preflop table) ──
                    const board = this.parseBoard();
                    const street = BOARD_LENGTH_TO_STREET[board.length] ?? "preflop";
                    if (hand.length === 2) {
                        if (board.length === 0) {
                            const preflopEq = this.equityEstimator.quickPreflopEquity(hand);
                            if (preflopEq !== null) {
                                console.log(`[Equity] Preflop lookup: ${(preflopEq * 100).toFixed(1)}% (${hand.join(" ")})`);
                            }
                        } else {
                            const eq = this.equityEstimator.estimateVsRandomRange(hand, board, 1);
                            const method = this.equityEstimator.getLastMethod();
                            console.log(
                                `[Equity] ${street} (${method}): win=${(eq.winRate * 100).toFixed(1)}% ` +
                                `tie=${(eq.tieRate * 100).toFixed(1)}% (${eq.trials > 0 ? eq.trials + " trials" : "exact"})`
                            );
                        }
                        if (board.length >= 3) {
                            const evalResult = this.handEvaluator.evaluate([...hand, ...board]);
                            console.log(`[Hand] ${evalResult.category} (rank: ${evalResult.rankValue})`);
                        }
                    }

                    await postProcessLogs(this.table.getLogsQueue(), this.game);

                    try {
                        this.table.resetPlayerActions();
                        const bot_action = await this.getEngineAction();
                        await this.performBotAction(bot_action);
                    } catch (err) {
                        console.log("Failed to compute or perform bot action:", err);
                    }

                    console.log("Waiting for bot's turn to end.");
                    logResponse(await this.puppeteer_service.waitForBotTurnEnd(), this.debug_mode);

                // ── WINNER DETECTED ────────────────────────────────────────
                } else if (data.includes("winner")) {
                    console.log("Detected winner in hand.");

                    try {
                        const endStack = await this.puppeteer_service.getStackSize();
                        if (endStack.code === "success") {
                            const trace        = this.local_poker_engine.getLastTrace();
                            const endStackBB   = convertToBBs(endStack.data as number, this.game.getBigBlind());
                            const hero         = this.game.getHero();
                            const startStackBB = hero?.getStackSize() ?? endStackBB;
                            const rawWonBB     = endStackBB - startStackBB;
                            const wonBB        = isNaN(rawWonBB) ? 0 : rawWonBB;

                            console.log(`[Game] Hand result: start=${startStackBB.toFixed(2)}BB end=${endStackBB.toFixed(2)}BB won=${wonBB.toFixed(2)}BB`);

                            const outcome: HandOutcome = {
                                handId:       `hand-${Date.now()}`,
                                street:       trace?.street       ?? "preflop",
                                heroPosition: trace?.position     ?? "unknown",
                                handKey:      trace?.handKey      ?? "??",
                                action:       trace?.chosenAction ?? "unknown",
                                sizeBB:       trace?.chosenSizeBB ?? 0,
                                potBB:        this.table.getPot(),
                                wonBB,
                                showdown:     wonBB !== 0,
                                villainVPIP:  undefined,
                                villainPFR:   undefined,
                            };
                            this.local_poker_engine.recordHandOutcome(outcome);
                        }
                    } catch (err) {
                        console.log("Failed to record hand outcome:", err);
                    }

                    await scrapeLog(this.puppeteer_service, this.villainMemory);

                    break;
                }
            }
        }

        const stackRes = await this.puppeteer_service.getStackSize();
        logResponse(stackRes, this.debug_mode);
        if (stackRes.code === "success") {
            console.log("Ending stack size:", stackRes.data);
        }

        try {
            processed_logs = await this.pullAndProcessLogs(
                this.first_created,
                processed_logs.first_fetch
            );
            await postProcessLogsAfterHand(processed_logs.valid_msgs, this.game);
            await this.table.processPlayers();
        } catch (err) {
            console.log("Failed to process players:", err);
        }
        
        logResponse(await this.puppeteer_service.waitForHandEnd(), this.debug_mode);
        console.log("Completed a hand.\n");
    }

    private async updateBoardState(): Promise<void> {
        const res = await this.puppeteer_service.getCommunityCards();

        if (res.code !== "success") {
            console.warn("[Board] Failed to scrape community cards — defaulting to preflop state.");
            this.table.setRunout("");
            this.table.setStreet("preflop");
            return;
        }

        const board_cards = res.data as string[];
        const street = BOARD_LENGTH_TO_STREET[board_cards.length] ?? "preflop";
        const runout_str = board_cards.length > 0
            ? `[${board_cards.join(", ")}]`
            : "";

        this.table.setRunout(runout_str);
        this.table.setStreet(street);

        console.log(`[Board] Street : ${street}`);
        console.log(`[Board] Runout : ${runout_str || "(none — preflop)"}`);
    }

    private async getEngineAction(): Promise<BotAction> {
        const heroId = this.game.getHero()?.getPlayerId() ?? "";
        const villainId = getVillainId(this.table, heroId);
        const memProfile = villainId ? this.villainMemory.getProfile(villainId) : null;
        const adj = getEngineAdjustments(memProfile);

        if (villainId && memProfile) {
            this.villainMemory.printHUD(villainId);
            console.log(
                `[Bot] Adjustments -> callD:${adj.callThresholdDelta} bluffD:${adj.bluffFreqDelta} 3bD:${adj.threeBetFreqDelta}`
            );
        }

        this.local_poker_engine.sessionTracker.applyVillainAdjustments(adj);

        const gtoProfile = selectProfile(
            memProfile?.vpip ?? 0, memProfile?.pfr ?? 0,
            memProfile?.totalHands ?? 0,
            convertToBBs(this.game.getHero()?.getStackSize() ?? 100, this.game.getBigBlind())
        );
        console.log(`[GTO] Profile: ${gtoProfile.name} — ${gtoProfile.description}`);

        if ((this.table.getStreet() ?? "preflop").toLowerCase() === "preflop" && villainId) {
            this.local_poker_engine.resetVillainRange(villainId);
        }

        // ══════════════════════════════════════════════════════════════════
        // DECISION PIPELINE: RLCard → Local Engine → LLM API
        // ══════════════════════════════════════════════════════════════════

        let action: BotAction | null = null;
        let source = "";

        // ── STEP 1: RLCard (primary decision maker) ──────────────────────
        if (this.rlcardAvailable) {
            try {
                const rlDecision = await this.consultRLCardFull();
                if (rlDecision) {
                    action = rlDecision.botAction;
                    source = "rlcard";

                    // Confidence-based bet sizing:
                    // High confidence (>60%) → scale up bet
                    // Very high confidence (>80%) → aggressive overbet
                    if (action.bet_size_in_BBs > 0 && ["bet", "raise"].includes(action.action_str)) {
                        const conf = rlDecision.winConfidence;
                        const potBB = this.table.getPot();
                        const stackBB = this.game.getHero()?.getStackSize() ?? 100;

                        if (conf >= 0.80) {
                            // Very high confidence → overbet 1.2x-1.5x pot
                            const aggressiveSize = Math.min(potBB * 1.5, stackBB);
                            action.bet_size_in_BBs = Math.max(action.bet_size_in_BBs, aggressiveSize);
                            console.log(`[RLCard] HIGH confidence ${(conf * 100).toFixed(0)}% → sizing up to ${action.bet_size_in_BBs.toFixed(2)}bb`);
                        } else if (conf >= 0.60) {
                            // Good confidence → bet 0.75x-1x pot
                            const solidSize = Math.min(potBB * 0.85, stackBB);
                            action.bet_size_in_BBs = Math.max(action.bet_size_in_BBs, solidSize);
                            console.log(`[RLCard] GOOD confidence ${(conf * 100).toFixed(0)}% → sizing to ${action.bet_size_in_BBs.toFixed(2)}bb`);
                        }
                    }

                    // If rlcard says fold/check but confidence is high, override to bet
                    if (action.action_str === "check" && rlDecision.winConfidence >= 0.65) {
                        const potBB = this.table.getPot();
                        const stackBB = this.game.getHero()?.getStackSize() ?? 100;
                        const betSize = Math.min(potBB * 0.6, stackBB);
                        if (betSize >= 1) {
                            console.log(`[RLCard] Confidence ${(rlDecision.winConfidence * 100).toFixed(0)}% but check → converting to bet ${betSize.toFixed(2)}bb`);
                            action = { action_str: "bet", bet_size_in_BBs: betSize };
                        }
                    }

                    console.log(
                        `[RLCard] Primary decision: ${action.action_str}` +
                        (action.bet_size_in_BBs > 0 ? ` ${action.bet_size_in_BBs.toFixed(2)}bb` : "") +
                        ` (confidence: ${(rlDecision.winConfidence * 100).toFixed(1)}%)`
                    );

                    if (await this.isValidBotAction(action)) {
                        console.log("[RLCard] Action validated against UI.");
                        this.local_poker_engine.sessionTracker.clearVillainAdjustments();
                        return action;
                    }
                    console.log("[RLCard] Action not valid in UI — falling through to engine.");
                    action = null;
                }
            } catch (err) {
                console.warn("[RLCard] Primary decision failed:", (err as Error).message);
                action = null;
            }
        }

        // ── STEP 2: Local Poker Engine (secondary) ───────────────────────
        console.log("[Engine] Using local poker engine.");
        try {
            action = this.local_poker_engine.decide(this.game, this.table);
            source = "engine";
            console.log(
                `[Engine] Decision: ${action.action_str}` +
                (action.bet_size_in_BBs > 0 ? ` ${action.bet_size_in_BBs.toFixed(2)}bb` : "")
            );
        } catch (err) {
            console.warn("[Engine] Engine threw during decide():", err);
            action = defaultFoldAction;
            source = "engine-fallback";
        } finally {
            this.local_poker_engine.sessionTracker.clearVillainAdjustments();
        }

        // Log the full decision trace
        const trace = this.local_poker_engine.getLastTrace();
        if (trace) {
            console.log(
                `[Engine] ${trace.street.toUpperCase()} | ${trace.position} | ` +
                `${trace.handKey} -> ${trace.chosenAction}` +
                `${trace.chosenSizeBB > 0 ? ` ${trace.chosenSizeBB.toFixed(1)}bb` : ""}` +
                ` [profile: ${trace.profile}]` +
                ` eq: ${trace.equity?.toFixed(3) ?? "?"}` +
                ` spr: ${trace.spr?.toFixed(1) ?? "?"}` +
                ` pot-odds: ${trace.potOdds?.toFixed(3) ?? "?"}` +
                ` | reasons: ${trace.reasons.join(", ")}`
            );
        }

        // ── SEE-THE-FLOP OVERRIDE (preflop only) ─────────────────────────
        const SEE_FLOP_MAX_CALL_BB = 4;

        if (action.action_str === "fold" && (trace?.street ?? "preflop") === "preflop") {
            if (await this.isValidBotAction(defaultCheckAction)) {
                console.log("[Engine] See-flop override: check is free — taking it instead of folding.");
                return defaultCheckAction;
            }

            const heroHand = this.game.getHero()?.getHand() ?? [];
            const lookupEq = heroHand.length === 2
                ? this.equityEstimator.quickPreflopEquity(heroHand)
                : null;
            const equity        = lookupEq ?? trace?.equity ?? 0;
            const potOdds       = trace?.potOdds ?? 1;
            const potBB         = this.table.getPot();
            const impliedCallBB = potOdds < 1 ? (potBB * potOdds) / (1 - potOdds) : Infinity;

            const worthCalling =
                equity >= potOdds &&
                impliedCallBB > 0 &&
                impliedCallBB <= SEE_FLOP_MAX_CALL_BB;

            if (worthCalling) {
                const callAction: BotAction = { action_str: "call", bet_size_in_BBs: 0 };
                if (await this.isValidBotAction(callAction)) {
                    console.log(
                        `[Engine] See-flop override: equity ${(equity * 100).toFixed(1)}% >= ` +
                        `pot-odds ${(potOdds * 100).toFixed(1)}% | ` +
                        `implied call ~${impliedCallBB.toFixed(1)}bb <= ${SEE_FLOP_MAX_CALL_BB}bb max -> calling.`
                    );
                    return callAction;
                }
            }
        }

        // ── STEP 3: LLM API (last resort, when engine says fold) ─────────
        if (action.action_str === "fold" && this.aiService) {
            try {
                const aiOverride = await this.consultAI();
                if (aiOverride && aiOverride.action_str !== "fold") {
                    console.log(
                        `[AI] Override: engine=fold -> AI=${aiOverride.action_str}` +
                        (aiOverride.bet_size_in_BBs > 0 ? ` ${aiOverride.bet_size_in_BBs.toFixed(2)}bb` : "")
                    );
                    if (await this.isValidBotAction(aiOverride)) {
                        return aiOverride;
                    }
                    console.log("[AI] AI suggestion not valid in UI, keeping engine fold.");
                } else {
                    console.log("[AI] AI agrees with fold.");
                }
            } catch (err) {
                console.warn("[AI] Consultation failed:", (err as Error).message);
            }
        }

        // ── Validate and return ──────────────────────────────────────────
        if (await this.isValidBotAction(action)) {
            console.log(`[${source}] Action validated against UI.`);
            return action;
        }

        console.warn(`[${source}] Proposed action not available in UI. Trying check.`);
        if (await this.isValidBotAction(defaultCheckAction)) return defaultCheckAction;

        console.log("[Bot] Check not available. Falling back to fold.");
        return defaultFoldAction;
    }

    /**
     * Ask the AI for a poker decision using the full game state.
     * Returns null if the AI can't produce a valid action.
     */
    private async consultAI(): Promise<BotAction | null> {
        if (!this.aiService) return null;

        const query = constructQuery(this.game);
        console.log("[AI] Sending query to AI...");

        const response = await this.aiService.query(query, this.aiMessages);

        // Keep conversation history for multi-street context
        this.aiMessages = [
            ...response.prev_messages,
            response.curr_message
        ];

        const aiAction = response.bot_action;
        if (!aiAction.action_str) return null;

        console.log(
            `[AI] Response: ${aiAction.action_str}` +
            (aiAction.bet_size_in_BBs > 0 ? ` ${aiAction.bet_size_in_BBs.toFixed(2)}bb` : "")
        );

        return aiAction;
    }

    /**
     * Ask the rlcard Python agent for a decision based on the current game state.
     * Returns null if unavailable.
     */
    private async consultRLCard(): Promise<BotAction | null> {
        if (!this.rlcardAvailable) return null;

        const hero = this.game.getHero();
        if (!hero) return null;

        const hand = hero.getHand() ?? [];
        const bigBlind = this.game.getBigBlind();
        const board = this.getBoard();
        const heroStackBB = hero.getStackSize();
        const potBB = bigBlind > 0 ? this.table.getPot() / bigBlind : this.table.getPot();
        const heroId = hero.getPlayerId();

        // Get villain stack
        let villainStackBB = 100;
        for (const [id, stack] of this.table.getPlayerInitialStacks().entries()) {
            if (id !== heroId) {
                villainStackBB = bigBlind > 0 ? (stack as number) / bigBlind : (stack as number);
                break;
            }
        }

        // Get facing bet from last action
        const actions = this.table.getPlayerActions();
        let facingBetBB = 0;
        for (let i = actions.length - 1; i >= 0; i--) {
            const act = actions[i].getAction().toLowerCase();
            if (act === "bet" || act === "raise") {
                facingBetBB = bigBlind > 0 ? actions[i].getBetAmount() / bigBlind : actions[i].getBetAmount();
                break;
            }
            if (act === "check") break;
        }

        const street = (this.table.getStreet() ?? "preflop").toLowerCase();

        console.log(`[RLCard] Consulting rlcard: ${hand.join(",")} | ${board.join(",") || "preflop"} | pot:${potBB.toFixed(1)}bb`);

        try {
            return await this.rlcardService.decide({
                hand,
                board,
                pot_bb: potBB,
                hero_stack_bb: heroStackBB,
                villain_stack_bb: villainStackBB,
                facing_bet_bb: facingBetBB,
                street,
                big_blind: bigBlind,
            });
        } catch (err) {
            console.warn("[RLCard] Decision failed:", (err as Error).message);
            // Mark as unavailable for the rest of this session to avoid repeated timeouts
            this.rlcardAvailable = false;
            return null;
        }
    }

    /**
     * Like consultRLCard but returns the full decision including confidence
     * and action probabilities for the restructured pipeline.
     */
    private async consultRLCardFull(): Promise<RLCardFullDecision | null> {
        if (!this.rlcardAvailable) return null;

        const hero = this.game.getHero();
        if (!hero) return null;

        const hand = hero.getHand() ?? [];
        const bigBlind = this.game.getBigBlind();
        const board = this.getBoard();
        const heroStackBB = hero.getStackSize();
        const potBB = bigBlind > 0 ? this.table.getPot() / bigBlind : this.table.getPot();
        const heroId = hero.getPlayerId();

        let villainStackBB = 100;
        for (const [id, stack] of this.table.getPlayerInitialStacks().entries()) {
            if (id !== heroId) {
                villainStackBB = bigBlind > 0 ? (stack as number) / bigBlind : (stack as number);
                break;
            }
        }

        const actions = this.table.getPlayerActions();
        let facingBetBB = 0;
        for (let i = actions.length - 1; i >= 0; i--) {
            const act = actions[i].getAction().toLowerCase();
            if (act === "bet" || act === "raise") {
                facingBetBB = bigBlind > 0 ? actions[i].getBetAmount() / bigBlind : actions[i].getBetAmount();
                break;
            }
            if (act === "check") break;
        }

        const street = (this.table.getStreet() ?? "preflop").toLowerCase();
        console.log(`[RLCard] Consulting rlcard: ${hand.join(",")} | ${board.join(",") || "preflop"} | pot:${potBB.toFixed(1)}bb`);

        try {
            return await this.rlcardService.decideWithProbs({
                hand,
                board,
                pot_bb: potBB,
                hero_stack_bb: heroStackBB,
                villain_stack_bb: villainStackBB,
                facing_bet_bb: facingBetBB,
                street,
                big_blind: bigBlind,
            });
        } catch (err) {
            console.warn("[RLCard] Full decision failed:", (err as Error).message);
            this.rlcardAvailable = false;
            return null;
        }
    }

    private getBoard(): string[] {
        const runout = this.table.getRunout();
        if (!runout) return [];
        return runout.replace(/[\[\]]/g, "").split(",").map((s: string) => s.trim()).filter(Boolean);
    }

    private async updateGameInfo() {
        logResponse(await this.puppeteer_service.waitForGameInfo(), this.debug_mode);
    
        console.log("Getting game info.");
        const res = await this.puppeteer_service.getGameInfo();
        logResponse(res, this.debug_mode);
        if (res.code == "success") {
            const game_info = this.puppeteer_service.convertGameInfo(res.data as string);
            console.log(`[Game] Blinds updated — SB: ${game_info.small_blind} / BB: ${game_info.big_blind}`);
            this.game.updateGameTypeAndBlinds(
                game_info.small_blind, game_info.big_blind, game_info.game_type
            );
        } else {
            throw new Error("Failed to get game info.");
        }
    }

    private async pullAndProcessLogs(
        last_created: string,
        first_fetch: boolean
    ): Promise<ProcessedLogs> {
        const log = await this.log_service.fetchData("", last_created);
        if (log.code === "success") {
            let data = this.log_service.getData(log);
            let msg  = this.log_service.getMsg(data);

            if (first_fetch) {
                data = this.log_service.pruneLogsBeforeCurrentHand(data);
                msg  = this.log_service.getMsg(data);
                this.table.setPlayerInitialStacksFromMsg(msg, this.game.getBigBlind());

                first_fetch = false;
                this.first_created = this.log_service.getLast(
                    this.log_service.getCreatedAt(data)
                );

                const stack_msg   = getPlayerStacksMsg(msg);
                const id_to_stack = getIdToInitialStackFromMsg(stack_msg, this.game.getBigBlind());
                const seat_to_id  = getTableSeatToIdFromMsg(stack_msg);
                const id_to_seat  = getIdToTableSeatFromMsg(stack_msg);
                const id_to_name  = getIdToNameFromMsg(stack_msg);
                const name_to_id  = getNameToIdFromMsg(stack_msg);

                this.table.setIdToStack(id_to_stack);
                this.table.setTableSeatToId(seat_to_id);
                this.table.setIdToTableSeat(id_to_seat);
                this.table.setIdToName(id_to_name);
                this.table.setNameToId(name_to_id);

                await this.table.updateCache();
            }

            const only_valid = validateAllMsg(msg);
            preProcessLogs(only_valid, this.game);

            const first_seat_number = this.table.getSeatNumberFromId(
                this.table.getFirstSeatOrderId()
            );
            this.table.setIdToPosition(first_seat_number);
            this.table.convertAllOrdersToPosition();

            last_created = this.log_service.getFirst(
                this.log_service.getCreatedAt(data)
            );

            return {
                valid_msgs: only_valid,
                last_created,
                first_fetch
            };
        } else {
            throw new Error("Failed to pull logs.");
        }
    }

    private parseBoard(): string[] {
        const runout = this.table.getRunout();
        if (!runout) return [];
        return runout.replace(/[\[\]]/g, "").split(",").map(s => s.trim()).filter(Boolean);
    }

    private async getPotSize(): Promise<number> {
        let pot_size: number = 0;
        const res = await this.puppeteer_service.getPotSize();
        logResponse(res, this.debug_mode);
        if (res.code === "success") {
            pot_size = res.data as number;
        }
        return pot_size;
    }

    private async getHand(): Promise<string[]> {
        let hand: string[] = [];
        const res = await this.puppeteer_service.getHand();
        logResponse(res, this.debug_mode);
        if (res.code === "success") {
            hand = res.data as string[];
        }
        return hand;
    }

    private async getStackSize(): Promise<number> {
        let stack_size: number = 0;
        const res = await this.puppeteer_service.getStackSize();
        logResponse(res, this.debug_mode);
        if (res.code === "success") {
            stack_size = res.data as number;
        }
        return stack_size;
    }

    private async updateHero(hand: string[], stack_size: number): Promise<void> {
        const hero = this.game.getHero();
        if (!hero) {
            this.game.createAndSetHero(
                this.table.getIdFromName(this.bot_name), hand, stack_size
            );
        } else {
            hero.setHand(hand);
            hero.setStackSize(stack_size);
        }
    }

    private async isValidBotAction(bot_action: BotAction): Promise<boolean> {
        console.log("Attempted Bot Action:", bot_action);
        const valid_actions: string[] = ["bet", "raise", "call", "check", "fold", "all-in"];
        const curr_stack_size_in_BBs = this.game.getHero()!.getStackSize();
        console.log("Bot Stack in BBs:", curr_stack_size_in_BBs);

        let is_valid = false;
        if (bot_action.action_str && valid_actions.includes(bot_action.action_str)) {
            let res;
            switch (bot_action.action_str) {
                case "bet":
                    res = await this.puppeteer_service.waitForBetOption();
                    if (res.code === "success" &&
                        bot_action.bet_size_in_BBs > 0 &&
                        bot_action.bet_size_in_BBs <= curr_stack_size_in_BBs) {
                        is_valid = true;
                    }
                    break;
                case "raise":
                    res = await this.puppeteer_service.waitForBetOption();
                    if (res.code === "success" &&
                        bot_action.bet_size_in_BBs > 0 &&
                        bot_action.bet_size_in_BBs <= curr_stack_size_in_BBs) {
                        is_valid = true;
                    }
                    break;
                case "all-in":
                    res = await this.puppeteer_service.waitForBetOption();
                    if (res.code === "success") {
                        is_valid = true;
                    }
                    break;
                case "call":
                    // ── FIX: call does not require bet_size_in_BBs > 0.
                    // The UI call button handles the amount — the engine
                    // always passes 0 for calls (the amount is implicit).
                    res = await this.puppeteer_service.waitForCallOption();
                    if (res.code === "success") {
                        is_valid = true;
                    }
                    break;
                case "check":
                    res = await this.puppeteer_service.waitForCheckOption();
                    if (res.code === "success" && bot_action.bet_size_in_BBs == 0) {
                        is_valid = true;
                    }
                    break;
                case "fold":
                    res = await this.puppeteer_service.waitForFoldOption();
                    if (res.code === "success" && bot_action.bet_size_in_BBs == 0) {
                        is_valid = true;
                    }
                    break;
            }
        }
        return is_valid;
    }

    private async performBotAction(bot_action: BotAction): Promise<void> {
        console.log("Bot Action:", bot_action.action_str);
        let bet_size = convertToValue(bot_action.bet_size_in_BBs, this.game.getBigBlind());
        console.log("Bot Action (BBs):", bot_action);
        console.log("Big Blind:", this.game.getBigBlind());
        console.log("Converted Bet Size:", bet_size);
        console.log("Current Pot (BB):", this.table.getPot());
        console.log("Current Street:", this.table.getStreet());

        switch (bot_action.action_str) {
            case "bet":
                console.log("Bet Size:", convertToBBs(bet_size, this.game.getBigBlind()));
                logResponse(await this.puppeteer_service.betOrRaise(bet_size), this.debug_mode);
                break;
            case "raise":
                console.log("Bet Size:", convertToBBs(bet_size, this.game.getBigBlind()));
                logResponse(await this.puppeteer_service.betOrRaise(bet_size), this.debug_mode);
                break;
            case "all-in":
                bet_size = convertToValue(
                    this.game.getHero()!.getStackSize(), this.game.getBigBlind()
                );
                console.log("Bet Size:", convertToBBs(bet_size, this.game.getBigBlind()));
                logResponse(await this.puppeteer_service.betOrRaise(bet_size), this.debug_mode);
                break;
            case "call":
                logResponse(await this.puppeteer_service.call(), this.debug_mode);
                break;
            case "check":
                logResponse(await this.puppeteer_service.check(), this.debug_mode);
                break;
            case "fold":
                logResponse(await this.puppeteer_service.fold(), this.debug_mode);
                const res = await this.puppeteer_service.cancelUnnecessaryFold();
                if (res.code === "success") {
                    logResponse(await this.puppeteer_service.check(), this.debug_mode);
                }
                break;
        }
    }
}

async function scrapeLog(
    puppeteerService: PuppeteerService,
    villainMemory: VillainMemory
): Promise<void> {
    try {
        const logText: string = await puppeteerService.getPage().evaluate(() => {
            const selectors = [
                ".chat-messages-list",
                ".log-messages-list",
                ".game-log-container",
            ];
            const texts: string[] = [];
            const root = globalThis as any;
            for (const selector of selectors) {
                const elements = root.document?.querySelectorAll(selector) ?? [];
                for (const element of elements) {
                    texts.push((element as any).innerText ?? "");
                }
            }
            return texts.join("\n");
        });

        if (logText.trim().length > 0) {
            villainMemory.ingestLog(logText);
        }
    } catch (err) {
        console.warn("[Bot] Log scrape failed:", (err as Error).message);
    }
}

function getVillainId(table: Table, heroId: string): string | null {
    for (const [id] of table.getIdToName().entries()) {
        if (id !== heroId) return id;
    }
    return null;
}

export { scrapeLog, getVillainId };

// VERSION 2.0.1 — call validation fix, blind debug logging