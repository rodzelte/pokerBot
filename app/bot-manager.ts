import dotenv from 'dotenv';
import prompt from 'prompt-sync';
import { Bot } from './bot.ts'
import ai_config_json from './configs/ai-config.json' with { type: "json" };
import bot_config_json from './configs/bot-config.json' with { type: "json" };
import webdriver_config_json from './configs/webdriver-config.json' with { type: "json" };
import { DBService } from './services/db-service.ts';
import { LogService } from './services/log-service.ts';
import { PlayerService } from './services/player-service.ts';
import { PuppeteerService } from './services/puppeteer-service.ts';
import { AIConfig, BotConfig, WebDriverConfig } from './interfaces/config-interfaces.ts';
import { AIServiceFactory } from './helpers/ai-service-factory.ts';

const io = prompt();
const ai_config: AIConfig = ai_config_json;
const bot_config: BotConfig = bot_config_json;
const webdriver_config: WebDriverConfig = webdriver_config_json;

function init(): string {
    dotenv.config();
    return io("Enter the PokerNow game id (ex. https://www.pokernow.club/games/{game_id}): ");
}

const bot_manager = async function() {
    const game_id = init();

    const puppeteer_service = new PuppeteerService(
        webdriver_config.default_timeout,
        webdriver_config.headless_flag
    );
    await puppeteer_service.init();

    const db_service = new DBService("./app/pokernow-gpt.db");
    await db_service.init();

    const player_service = new PlayerService(db_service);

    const log_service = new LogService(game_id);
    await log_service.init();

    // Create AI service (optional — used as second opinion on borderline folds)
    let ai_service = null;
    try {
        const factory = new AIServiceFactory();
        ai_service = factory.createAIService(
            ai_config.provider, ai_config.model_name, ai_config.playstyle
        );
        ai_service.init();
        console.log(`[Bot] AI connected — ${ai_config.provider}/${ai_config.model_name} (${ai_config.playstyle})`);
    } catch (err) {
        console.warn("[Bot] AI service unavailable — running on local engine only.", (err as Error).message);
    }

    const bot = new Bot(
        log_service,
        player_service,
        puppeteer_service,
        game_id,
        bot_config.debug_mode,
        ai_service
    );

    await bot.run();
}

export default bot_manager;