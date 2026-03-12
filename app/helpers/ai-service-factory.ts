import { AIService } from "../interfaces/ai-client-interfaces.ts";
import { GoogleAIService } from "../services/ai/googleai-service.ts";
import { OpenAIService } from "../services/ai/openai-service.ts";
import { RLCardService } from "../services/ai/rlcard-service.ts";

export class AIServiceFactory {
    private supportedModels: Map<string, string[]>;

    constructor(){
        this.supportedModels = new Map<string, string[]>([
            ["OpenAI", ["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o"]],
            ["Google", ["gemini-2.5-pro", "gemini-3-flash-preview", "gemini-3.1-pro-preview"]],
            ["RLCard", ["dqn", "nfsp", "cfr", "cfr_plus", "mccfr", "deep_cfr"]]
        ]);
    }

    createAIService(provider: string, model_name: string, playstyle: string = "neutral"): AIService {
        if (!(this.supportedModels.has(provider) && this.supportedModels.get(provider)!.includes(model_name))) {
            throw new Error("AI provider or model not supported, please check the list of supported models.")
        }
        switch(provider) {
            case ("OpenAI"):
                const openai_auth_key = process.env.OPENAI_API_KEY;
                if (!openai_auth_key) {
                    throw new Error(`Invalid ${provider} auth key.`);
                }
                return new OpenAIService(openai_auth_key, model_name, playstyle);
            case ("Google"):
                const googleai_auth_key = process.env.GOOGLEAI_API_KEY;
                if (!googleai_auth_key) {
                    throw new Error (`Invalid ${provider} auth key.`);
                }
                return new GoogleAIService(googleai_auth_key, model_name, playstyle);
            case ("RLCard"):
                const serverUrl = process.env.RLCARD_SERVER_URL ?? "http://127.0.0.1:5050";
                return new RLCardService(serverUrl, playstyle);
        }
        throw new Error("Failed to create AI service.");
    }

    printSupportedModels(): void {
        console.log("Supported models:")
        for (const provider of Array.from(this.supportedModels.keys())) {
            let out = provider.concat(": ");
            out = out.concat(this.supportedModels.get(provider)!.join(", "));
            console.log(out);
        }
    }
}