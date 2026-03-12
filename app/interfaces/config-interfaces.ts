import { DebugMode } from "../utils/error-handling-utils.ts"
import type { AlgorithmType, SelectionMode } from "./game-state-interfaces.ts"

export interface AIConfig {
    provider: string,
    model_name: string,
    playstyle: string,
    /** Algorithm selection and ensemble configuration */
    algorithm?: AlgorithmConfig
}

export interface AlgorithmConfig {
    /** How to select/combine algorithms: "single", "ensemble", "cascade" */
    selection_mode: SelectionMode,
    /** Primary algorithm (used in single mode) */
    primary: AlgorithmType,
    /** Algorithms to use (in priority order for cascade, weighted for ensemble) */
    algorithms: AlgorithmType[],
    /** Weights for ensemble mode (must sum to 1, same order as algorithms) */
    ensemble_weights?: number[],
    /** Minimum confidence to accept a decision in cascade mode (0-1) */
    cascade_min_confidence?: number,
    /** Per-algorithm parameters */
    params: AlgorithmParamsConfig
}

export interface AlgorithmParamsConfig {
    cfr?: {
        game: string,           // "limit-holdem" | "leduc-holdem"
        model_path: string
    },
    cfr_plus?: {
        game: string,
        model_path: string
    },
    mccfr?: {
        game: string,
        model_path: string
    },
    deep_cfr?: {
        game: string,           // "no-limit-holdem"
        model_path: string,
        device: string
    },
    dqn?: {
        epsilon: number,        // exploration rate for inference (typically 0)
        mlp_layers: number[],
        model_path: string,
        device: string
    },
    nfsp?: {
        anticipatory_param: number,
        hidden_layers: number[],
        q_mlp_layers: number[],
        evaluate_with: string,  // "average_policy" | "best_response"
        model_path: string,
        device: string
    }
}

export interface BotConfig {
    debug_mode: DebugMode,
    query_retries: number
}

export interface WebDriverConfig {
    default_timeout: number,
    headless_flag: boolean
}