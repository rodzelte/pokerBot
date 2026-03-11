/**
 * rlcard-service.ts
 *
 * TypeScript client that talks to the Python rlcard FastAPI server.
 * Implements AIService so it plugs into the existing bot architecture
 * (factory, second-opinion pipeline, etc.).
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

// ── Types matching the Python server's request / response ────────────────

interface RLCardGameState {
  hand: string[];
  board: string[];
  pot_bb: number;
  hero_stack_bb: number;
  villain_stack_bb: number;
  facing_bet_bb: number;
  street: string;
  big_blind: number;
}

interface RLCardDecision {
  action: string;
  size_bb: number;
  sizing_type: string;
  raw_action_id: number;
  action_probs: Record<string, number>;
}

/** Full decision including confidence metrics for the bot pipeline */
export interface RLCardFullDecision {
  botAction: BotAction;
  winConfidence: number;       // 0-1, how confident the model is in aggressive play
  actionProbs: Record<string, number>;
  sizingType: string;
}

// ── Service ──────────────────────────────────────────────────────────────

export class RLCardService extends AIService {
  private serverUrl: string;
  private healthy = false;

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
   */
  async isHealthy(): Promise<boolean> {
    try {
      const res = await fetch(`${this.serverUrl}/health`, {
        signal: AbortSignal.timeout(3000),
      });
      if (res.ok) {
        this.healthy = true;
        return true;
      }
    } catch {
      // server not reachable
    }
    this.healthy = false;
    return false;
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
      `[RLCard] ${decision.action}` +
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
    };
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
