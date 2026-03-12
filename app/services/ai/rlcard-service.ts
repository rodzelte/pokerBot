/**
 * rlcard-service.ts
 *
 * TypeScript client that talks to the Python rlcard FastAPI server.
 * Implements AIService so it plugs into the existing bot architecture
 * (factory, second-opinion pipeline, etc.).
 *
 * Supports multiple algorithm types (DQN, NFSP, CFR, CFR+, MCCFR, Deep CFR)
 * and can hot-switch between them at runtime, request ensemble decisions,
 * and accept full GameStateSnapshot payloads.
 *
 * The Python server must be running:
 *   cd python && pip install -r requirements.txt && python rlcard_server.py
 */
import {
  AIMessage,
  AIResponse,
  AIService,
  BotAction,
} from "../../interfaces/ai-client-interfaces.ts";
import type {
  AlgorithmType,
  AlgorithmDecision,
  GameStateSnapshot,
  EnsembleConfig,
} from "../../interfaces/game-state-interfaces.ts";

// ── Types matching the Python server's request / response ────────────────

export interface RLCardGameState {
  hand: string[];
  board: string[];
  pot_bb: number;
  hero_stack_bb: number;
  villain_stack_bb: number;
  facing_bet_bb: number;
  street: string;
  big_blind: number;
  /** Hero's table position (BTN, SB, BB, UTG, etc.) for position-aware decisions */
  hero_position?: string;
  /** Optional: full player info for multi-player awareness */
  players?: Array<{
    player_id: string;
    position: string;
    stack_bb: number;
    is_hero: boolean;
  }>;
  /** Optional: full action history */
  action_history?: Array<{
    player_id: string;
    action: string;
    amount_bb: number;
    street: string;
  }>;
}

interface RLCardDecision {
  action: string;
  size_bb: number;
  sizing_type: string;
  raw_action_id: number;
  action_probs: Record<string, number>;
}

/** Agent type supported by the rlcard Python server */
export type RLCardAgentType = "dqn" | "nfsp" | "cfr" | "cfr_plus" | "mccfr" | "deep_cfr";

/** Info returned by the /agent endpoint */
interface RLCardAgentInfo {
  agent_type: RLCardAgentType;
  agent_loaded: boolean;
  supported_types: RLCardAgentType[];
  available_agents?: RLCardAgentType[];
}

/** Full decision including confidence metrics for the bot pipeline */
export interface RLCardFullDecision {
  botAction: BotAction;
  winConfidence: number;       // 0-1, how confident the model is in aggressive play
  actionProbs: Record<string, number>;
  sizingType: string;
  source: RLCardAgentType;
}

/** Ensemble decision from the /ensemble endpoint */
interface RLCardEnsembleResponse {
  decisions: Array<{
    agent_type: string;
    action: string;
    size_bb: number;
    sizing_type: string;
    raw_action_id: number;
    action_probs: Record<string, number>;
    confidence: number;
  }>;
  combined: {
    action: string;
    size_bb: number;
    sizing_type: string;
    confidence: number;
    action_probs: Record<string, number>;
  };
}

// ── Service ──────────────────────────────────────────────────────────────

export class RLCardService extends AIService {
  private serverUrl: string;
  private healthy = false;
  private agentType: RLCardAgentType = "dqn";
  private availableAgents: RLCardAgentType[] = [];

  /**
   * @param serverUrl  Base URL of the Python rlcard server (e.g. "http://127.0.0.1:5050")
   * @param playstyle  Passed to AIService base (unused by rlcard, kept for interface compat)
   */
  constructor(serverUrl?: string, playstyle: string = "neutral") {
    // api_key and model_name are not meaningful for rlcard, but AIService requires them
    super("rlcard", "dqn", playstyle);
    this.serverUrl = serverUrl ?? process.env.RLCARD_SERVER_URL ?? "http://127.0.0.1:5050";
  }

  init(): void {
    console.log(`[RLCard] Service initialised — server: ${this.serverUrl}`);
  }

  /**
   * Health-check: returns true if the Python server is reachable.
   * Also caches the agent type and available agents reported by the server.
   */
  async isHealthy(): Promise<boolean> {
    try {
      const res = await fetch(`${this.serverUrl}/health`, {
        signal: AbortSignal.timeout(3000),
      });
      if (res.ok) {
        const data = await res.json() as {
          status: string;
          agent_type?: string;
          available_agents?: string[];
        };
        this.healthy = true;
        if (data.agent_type) {
          this.agentType = data.agent_type as RLCardAgentType;
        }
        if (data.available_agents) {
          this.availableAgents = data.available_agents as RLCardAgentType[];
        }
        return true;
      }
    } catch {
      // server not reachable
    }
    this.healthy = false;
    return false;
  }

  /** Returns the agent type currently running on the Python server. */
  getAgentType(): RLCardAgentType {
    return this.agentType;
  }

  /** Returns the list of agents available on the server. */
  getAvailableAgents(): RLCardAgentType[] {
    return this.availableAgents;
  }

  /** Query the /agent endpoint for detailed server-side agent info. */
  async getAgentInfo(): Promise<RLCardAgentInfo> {
    const res = await fetch(`${this.serverUrl}/agent`, {
      signal: AbortSignal.timeout(3000),
    });
    if (!res.ok) {
      throw new Error(`[RLCard] /agent returned ${res.status}`);
    }
    return await res.json() as RLCardAgentInfo;
  }

  /**
   * Hot-switch the active algorithm on the Python server.
   * Returns true if successful.
   */
  async switchAlgorithm(agentType: AlgorithmType, modelPath?: string): Promise<boolean> {
    try {
      const res = await fetch(`${this.serverUrl}/switch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          agent_type: agentType,
          model_path: modelPath,
        }),
        signal: AbortSignal.timeout(10000),
      });
      if (res.ok) {
        const data = await res.json() as { agent_type: string; status: string };
        this.agentType = data.agent_type as RLCardAgentType;
        console.log(`[RLCard] Switched to ${this.agentType} algorithm.`);
        return true;
      }
      const errText = await res.text().catch(() => "");
      console.warn(`[RLCard] Switch failed (${res.status}): ${errText}`);
      return false;
    } catch (err) {
      console.warn("[RLCard] Switch request failed:", (err as Error).message);
      return false;
    }
  }

  /**
   * Main decision call.  Sends game state to the Python rlcard agent and
   * returns a BotAction compatible with the rest of the bot.
   */
  async decide(state: RLCardGameState): Promise<BotAction> {
    const full = await this.decideWithProbs(state);
    return full.botAction;
  }

  /**
   * Full decision call — returns action + probabilities + confidence score.
   * Used by the bot pipeline for confidence-based bet sizing.
   */
  async decideWithProbs(state: RLCardGameState): Promise<RLCardFullDecision> {
    const res = await fetch(`${this.serverUrl}/decide`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(state),
      signal: AbortSignal.timeout(5000),
    });

    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`[RLCard] Server returned ${res.status}: ${text}`);
    }

    const decision: RLCardDecision = await res.json() as RLCardDecision;

    // Compute win confidence: sum of aggressive action probabilities
    // (raise_half_pot + raise_full_pot + raise_all_in)
    const probs = decision.action_probs;
    const aggressiveProb =
      (probs["raise_half_pot"] ?? 0) +
      (probs["raise_full_pot"] ?? 0) +
      (probs["raise_all_in"] ?? 0);
    const callProb = probs["call"] ?? 0;
    // Confidence = how much the model wants to put money in
    const winConfidence = Math.min(1, aggressiveProb + callProb * 0.5);

    console.log(
      `[RLCard:${this.agentType}] ${decision.action}` +
      (decision.size_bb > 0 ? ` ${decision.size_bb.toFixed(2)}bb` : "") +
      ` (${decision.sizing_type})` +
      ` | confidence: ${(winConfidence * 100).toFixed(1)}%` +
      ` | probs: ${JSON.stringify(probs)}`
    );

    return {
      botAction: {
        action_str: decision.action,
        bet_size_in_BBs: decision.size_bb,
      },
      winConfidence,
      actionProbs: probs,
      sizingType: decision.sizing_type,
      source: this.agentType,
    };
  }

  /**
   * Request a specific algorithm for this one decision (without permanently switching).
   * The Python server handles this via the `agent_type` field in the request.
   */
  async decideWithAlgorithm(
    state: RLCardGameState,
    algorithmType: AlgorithmType
  ): Promise<RLCardFullDecision> {
    const res = await fetch(`${this.serverUrl}/decide`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...state, agent_type: algorithmType }),
      signal: AbortSignal.timeout(5000),
    });

    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`[RLCard:${algorithmType}] Server returned ${res.status}: ${text}`);
    }

    const decision: RLCardDecision = await res.json() as RLCardDecision;
    const probs = decision.action_probs;
    const aggressiveProb =
      (probs["raise_half_pot"] ?? 0) +
      (probs["raise_full_pot"] ?? 0) +
      (probs["raise_all_in"] ?? 0);
    const callProb = probs["call"] ?? 0;
    const winConfidence = Math.min(1, aggressiveProb + callProb * 0.5);

    return {
      botAction: {
        action_str: decision.action,
        bet_size_in_BBs: decision.size_bb,
      },
      winConfidence,
      actionProbs: probs,
      sizingType: decision.sizing_type,
      source: algorithmType as RLCardAgentType,
    };
  }

  /**
   * Request an ensemble decision from all loaded algorithms.
   * The server queries each algorithm and returns weighted combination.
   */
  async decideEnsemble(
    state: RLCardGameState,
    config: EnsembleConfig
  ): Promise<RLCardFullDecision> {
    const res = await fetch(`${this.serverUrl}/ensemble`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...state,
        algorithms: config.algorithms,
        weights: config.weights,
      }),
      signal: AbortSignal.timeout(10000),
    });

    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`[RLCard:ensemble] Server returned ${res.status}: ${text}`);
    }

    const ensemble: RLCardEnsembleResponse = await res.json() as RLCardEnsembleResponse;

    // Log individual algorithm votes
    for (const d of ensemble.decisions) {
      console.log(
        `[RLCard:${d.agent_type}] vote: ${d.action}` +
        (d.size_bb > 0 ? ` ${d.size_bb.toFixed(2)}bb` : "") +
        ` (conf: ${(d.confidence * 100).toFixed(1)}%)`
      );
    }

    const combined = ensemble.combined;
    console.log(
      `[RLCard:ensemble] Combined: ${combined.action}` +
      (combined.size_bb > 0 ? ` ${combined.size_bb.toFixed(2)}bb` : "") +
      ` | confidence: ${(combined.confidence * 100).toFixed(1)}%`
    );

    return {
      botAction: {
        action_str: combined.action,
        bet_size_in_BBs: combined.size_bb,
      },
      winConfidence: combined.confidence,
      actionProbs: combined.action_probs,
      sizingType: combined.sizing_type,
      source: this.agentType,
    };
  }

  /**
   * Convert a GameStateSnapshot to the RLCardGameState format
   * expected by the Python server.
   */
  static snapshotToGameState(snapshot: GameStateSnapshot): RLCardGameState {
    // Find the main villain (highest stack non-hero active player)
    const villains = snapshot.players.filter(p => !p.isHero);
    const mainVillain = villains.sort((a, b) => b.stackBB - a.stackBB)[0];

    return {
      hand: snapshot.heroHand,
      board: snapshot.board,
      pot_bb: snapshot.potBB,
      hero_stack_bb: snapshot.hero.stackBB,
      villain_stack_bb: mainVillain?.stackBB ?? 100,
      facing_bet_bb: snapshot.facingBetBB,
      street: snapshot.street,
      big_blind: snapshot.rules.bigBlind,
      hero_position: snapshot.hero.position,
      players: snapshot.players.map(p => ({
        player_id: p.playerId,
        position: p.position,
        stack_bb: p.stackBB,
        is_hero: p.isHero,
      })),
      action_history: snapshot.actionHistory.map(a => ({
        player_id: a.playerId,
        action: a.action,
        amount_bb: a.amountBB,
        street: a.street,
      })),
    };
  }

  /**
   * High-level decision from a GameStateSnapshot.
   * Converts the snapshot and queries the server.
   */
  async decideFromSnapshot(snapshot: GameStateSnapshot): Promise<RLCardFullDecision> {
    const gameState = RLCardService.snapshotToGameState(snapshot);
    return this.decideWithProbs(gameState);
  }

  /**
   * Cascade decision: try each algorithm in order until one returns
   * a decision above the minimum confidence threshold.
   */
  async decideCascade(
    state: RLCardGameState,
    algorithms: AlgorithmType[],
    minConfidence: number
  ): Promise<RLCardFullDecision | null> {
    for (const algo of algorithms) {
      try {
        const decision = await this.decideWithAlgorithm(state, algo);
        console.log(
          `[RLCard:cascade] ${algo}: ${decision.botAction.action_str}` +
          ` confidence=${(decision.winConfidence * 100).toFixed(1)}%`
        );
        if (decision.winConfidence >= minConfidence) {
          console.log(`[RLCard:cascade] Accepted ${algo} decision (conf >= ${(minConfidence * 100).toFixed(0)}%)`);
          return decision;
        }
        console.log(`[RLCard:cascade] ${algo} below threshold, trying next...`);
      } catch (err) {
        console.warn(`[RLCard:cascade] ${algo} failed:`, (err as Error).message);
      }
    }
    // All algorithms below threshold — return the last one tried
    console.log("[RLCard:cascade] No algorithm met confidence threshold.");
    return null;
  }

  /**
   * Query the server for current strategy storage info.
   */
  async getStrategyInfo(): Promise<Record<string, unknown>> {
    try {
      const res = await fetch(`${this.serverUrl}/strategy`, {
        signal: AbortSignal.timeout(3000),
      });
      if (res.ok) {
        return await res.json() as Record<string, unknown>;
      }
    } catch {
      // not available
    }
    return {};
  }

  // ── AIService abstract method implementations ─────────────────────────

  /**
   * query() is the AIService interface method.
   * For rlcard we parse the game state from the query string (or use
   * the direct `decide()` method when called from the engine).
   */
  async query(input: string, prev_messages: AIMessage[]): Promise<AIResponse> {
    // Try to parse input as a JSON game state (when called directly)
    let state: RLCardGameState;
    try {
      state = JSON.parse(input);
    } catch {
      // If the input is a human-readable query (from constructQuery), use defaults
      state = {
        hand: [], board: [], pot_bb: 0,
        hero_stack_bb: 100, villain_stack_bb: 100,
        facing_bet_bb: 0, street: "preflop", big_blind: 1,
      };
    }

    const botAction = await this.decide(state);

    const respMessage: AIMessage = {
      text_content: `{${botAction.action_str} ${botAction.bet_size_in_BBs}}`,
      metadata: { role: "assistant" },
    };

    return {
      bot_action: botAction,
      prev_messages: [...prev_messages, { text_content: input, metadata: { role: "user" } }],
      curr_message: respMessage,
    };
  }

  processMessages(messages: AIMessage[]): any[] {
    return messages;
  }
}
