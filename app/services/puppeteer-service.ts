import puppeteer from 'puppeteer';

import { computeTimeout, sleep } from '../helpers/bot-helper.ts';

import type { Response } from '../utils/error-handling-utils.ts';

interface GameInfo {
    game_type: string,
    big_blind: number,
    small_blind: number,
}

export class PuppeteerService {
    private default_timeout: number;
    private headless_flag: boolean;

    private browser!: puppeteer.Browser;
    private page!: puppeteer.Page;

    constructor(default_timeout: number, headless_flag: boolean) {
        this.default_timeout = default_timeout;
        this.headless_flag = headless_flag;
    }

    async init(): Promise<void> {
        this.browser = await puppeteer.launch({
            defaultViewport: null,
            headless: this.headless_flag
        });
        this.page = await this.browser.newPage();
    }

    async closeBrowser(): Promise<void> {
        await this.browser.close();
    }

    public getPage(): puppeteer.Page {
        return this.page;
    }
    
    async navigateToGame<D, E=Error>(game_id: string): Response<D, E> {
        if (!game_id) {
            return {
                code: "error",
                error: new Error("Game id cannot be empty.") as E
            }
        }
        await this.page.goto(`https://www.pokernow.club/games/${game_id}`);
        await this.page.setViewport({width: 1024, height: 768});
        return {
            code: "success",
            data: null as D,
            msg: `Successfully opened PokerNow game with id ${game_id}.`
        }
    }
    
    async waitForGameInfo<D, E=Error>(): Response<D, E> {
        try {
            await this.page.waitForSelector('.game-infos > .blind-value-ctn > .blind-value', {timeout: this.default_timeout * 30});
        } catch (err) {
            return {
                code: "error",
                error: new Error("Failed to wait for game information.") as E
            }
        }
        return {
            code: "success",
            data: null as D,
            msg: "Successfully waited for game information."
        }
    }
    
    async getGameInfo<D, E=Error>(): Response<D, E> {
        var game_info;
        try {
            game_info = await this.page.$eval(".game-infos > .blind-value-ctn > .blind-value", (div: any) => div.textContent);
        } catch (err) {
            return {
                code: "error",
                error: new Error("Could not get game info.") as E
            }
        }
        return {
            code: "success",
            data: game_info as D,
            msg: "Successfully grabbed the game info."
        }
    }
    
    convertGameInfo(game_info: string): GameInfo {
        const re = RegExp("([A-Z]+)~\\s([0-9]+)\\s\/\\s([0-9]+)");
        const matches = re.exec(game_info);
        if (matches && matches.length == 4) {
            return {game_type: matches[1], big_blind: Number(matches[3]), small_blind: Number(matches[2])};
        } else {
            throw new Error("Failed to convert game info.");
        }
    }
    
    // send enter table request as non-host player
    async sendEnterTableRequest<D, E=Error>(name: string, stack_size: number): Response<D, E> {
        if (name.length < 2 || name.length > 14) {
            return {
                code: "error",
                error: new Error("Player name must be betwen 2 and 14 characters long.") as E
            }
        }
        try {
            await this.page.waitForSelector(".table-player-seat-button", {timeout: this.default_timeout * 4});
            await this.page.$eval(".table-player-seat-button", (button: any) => button.click());
        } catch (err) {
            return {
                code: "error",
                error: new Error("Could not find open seat.") as E
            }
        }
        await this.page.focus(".selected > div > form > div:nth-child(1) > input");
        await this.page.keyboard.type(name);
        await this.page.focus(".selected > div > form > div:nth-child(2) > input");
        await this.page.keyboard.type(stack_size.toString())
        await this.page.$eval(".selected > div > form > button", (button: any) => button.click());
        try {
            await this.page.waitForSelector(".alert-1-buttons > button", {timeout: this.default_timeout});
            await this.page.$eval(".alert-1-buttons > button", (button: any) => button.click());
        } catch (err) {
            var message = "Table ingress unsuccessful."
            if (await this.page.$(".selected > div > form > div:nth-child(1) > .error-message")) {
                message = "Player name must be unique to game.";
            }
            await this.page.$eval(".selected > button", (button: any) => button.click());
            return {
                code: "error",
                error: new Error(message) as E
            }
        }
        return {
            code: "success",
            data: null as D,
            msg: "Table ingress request successfully sent."
        }
    }
    
    async waitForTableEntry<D, E=Error>(): Response<D, E> {
        try {
            await this.page.waitForSelector(".you-player", {timeout: this.default_timeout * 120});
        } catch (err) {
            return {
                code: "error",
                error: new Error("Table ingress request not accepted by host.") as E
            }
        }
        return {
            code: "success",
            data: null as D,
            msg: "Successfully entered table."
        }
    }
    
    async waitForNextHand<D, E=Error>(num_players: number, max_turn_length: number): Response<D, E> {
        try {
            await this.page.waitForSelector([".you-player > .waiting", ".you-player > .waiting-next-hand"].join(','), {timeout: this.default_timeout});
        } catch (err) {
            return {
                code: "error",
                error: new Error("Player is not in waiting state.") as E
            }
        }
        try {
            await this.page.waitForSelector([".you-player > .waiting", ".you-player > .waiting-next-hand"].join(','), 
            {hidden: true, timeout: computeTimeout(num_players, max_turn_length, 4) * 5 + this.default_timeout});
        } catch (err) {
            return {
                code: "error",
                error: new Error("Player is not in waiting state.") as E
            }
        }
        return {
            code: "success",
            data: null as D,
            msg: "Waited for next hand to start."
        }
    }
    
    async getNumPlayers<D, E=Error>(): Response<D, E> {
        try {
            await this.page.waitForSelector(".table-player", {timeout: this.default_timeout});
            const table_players_count = await this.page.$$eval(".table-player", (divs: any) => divs.length) as number;
            const table_player_status_count = await this.page.$$eval(".table-player-status-icon", (divs: any) => divs.length) as number;
            const num_players = table_players_count - table_player_status_count;
            return {
                code: "success",
                data: num_players as D,
                msg: `Successfully got number of players in table: ${num_players}`
            }
        } catch (err) {
            return {
                code: "error",
                error: new Error("Failed to compute number of players in table.") as E
            }
        }
    
    }
    
    async waitForBotTurnOrWinner<D, E=Error>(num_players: number, max_turn_length: number): Response<D, E> {
        try {
            const el = await this.page.waitForSelector([".action-signal", ".table-player.winner"].join(','), {timeout: computeTimeout(num_players, max_turn_length, 4) * 5 + this.default_timeout});
            const class_name = await this.page.evaluate(el => el!.className, el);
            return {
                code: "success",
                data: class_name as D,
                msg: `Waited for ${class_name}`
            }
        } catch (err) {
            return {
                code: "error",
                error: new Error("It is not the player's turn.") as E
            }
        }
    }
    
    async waitForBotTurnEnd<D, E=Error>(): Response<D, E> {
        try {
            await this.page.waitForSelector(".action-signal", {hidden: true, timeout: this.default_timeout * 15});
        } catch (err) {
            return {
                code: "error",
                error: new Error("Failed to wait for bot's turn to end.") as E
            }
        }
        return {
            code: "success",
            data: null as D,
            msg: "Successfully waited for bot's turn to end."
        }
    }
    
    async getPotSize<D, E=Error>(): Response<D, E> {
        try {
            await this.page.waitForSelector(".table > .table-pot-size > .main-value");
            const pot_size_str = await this.page.$eval(".table > .table-pot-size > .main-value", (p: any) => p.textContent);
            return {
                code: "success",
                data: pot_size_str as D,
                msg: "Successfully retrieved table pot size."
            }
        } catch (err) {
            return {
                code: "error",
                error: new Error("Failed to retrieve table pot size.") as E
            }
        }
    }
    
    async getHand<D, E=Error>(): Response<D, E> {
        try {
            const cards_div = await this.page.$$(".you-player > .table-player-cards > div");
            let cards: string[] = [];
            for (const card_div of cards_div) {
                const card_value = await card_div.$eval(".value", (span: any) => span.textContent);
                const sub_suit_letter = await card_div.$eval(".sub-suit", (span: any) => span.textContent);
                if (card_value && sub_suit_letter) {
                    cards.push(card_value + sub_suit_letter);
                } else {
                    throw "Invalid card.";
                }
            }
            return {
                code: "success",
                data: cards as D,
                msg: "Successfully retrieved player's hand."
            }
        } catch (err) {
            return {
                code: "error",
                error: new Error("Failed to retrieve player's hand.") as E
            }
        }
    }

    // ── NEW METHOD ──────────────────────────────────────────────────────────
    // Scrapes the community cards currently visible on the board.
    // Returns an empty array on preflop (no cards shown yet).
    // Selector confirmed against live PokerNow DOM:
    //   .table-cards .card-container  — each visible board card
    async getCommunityCards<D, E=Error>(): Response<D, E> {
        try {
            // If no board cards are present yet (preflop), the selector won't
            // match anything — that is valid, so we return an empty array.
            const card_divs = await this.page.$$(".table-cards .card-container");

            if (card_divs.length === 0) {
                console.log("[Puppeteer] No community cards on board (preflop).");
                return {
                    code: "success",
                    data: [] as unknown as D,
                    msg: "No community cards present."
                };
            }

            const cards: string[] = [];
            for (const card_div of card_divs) {
                // Each card container holds a .value span and a .sub-suit span,
                // matching the same structure used by getHand().
                const card_value      = await card_div.$eval(".value",    (s: any) => s.textContent).catch(() => null);
                const sub_suit_letter = await card_div.$eval(".sub-suit", (s: any) => s.textContent).catch(() => null);

                if (card_value && sub_suit_letter) {
                    cards.push(card_value + sub_suit_letter);
                } else {
                    console.warn("[Puppeteer] Skipping malformed community card div.");
                }
            }

            console.log("[Puppeteer] Community cards scraped:", cards.join(" "));
            return {
                code: "success",
                data: cards as unknown as D,
                msg: `Successfully scraped ${cards.length} community card(s).`
            };
        } catch (err) {
            return {
                code: "error",
                error: new Error("Failed to scrape community cards from board.") as E
            };
        }
    }
    // ── END NEW METHOD ──────────────────────────────────────────────────────

    async getStackSize<D, E=Error>(): Response<D, E> {
        try {
            await this.page.waitForSelector(".you-player > .table-player-infos-ctn > div > .table-player-stack");
            const stack_size_str = await this.page.$eval(".you-player > .table-player-infos-ctn > div > .table-player-stack", (p: any) => p.textContent);
            return {
                code: "success",
                data: stack_size_str as D,
                msg: "Successfully retrieved bot's stack size."
            }
        } catch (err) {
            return {
                code: "error",
                error: new Error("Failed to retrieve bot's stack size.") as E
            }
        }
    }

    async waitForCallOption<D, E=Error>(): Response<D, E> {
        try {
            await this.page.waitForSelector(".game-decisions-ctn > .action-buttons > .call", {timeout: this.default_timeout});
            const is_disabled = await this.page.$eval(".game-decisions-ctn > .action-buttons > .call", (button: any) => button.disabled);
            if (is_disabled) {
                throw new Error("Call option is disabled.")
            }
        } catch (err) {
            return {
                code: "error",
                error: new Error("No option to call available.") as E
            }
        }
        return {
            code: "success",
            data: null as D,
            msg: "Successfully waited for call option."
        }
    }
    
    async call<D, E=Error>(): Response<D, E> {
        try {
            await this.page.$eval(".game-decisions-ctn > .action-buttons > .call", (button: any) => button.click());
        } catch (err) {
            return {
                code: "error",
                error: new Error("Failed to execute call action.") as E
            }
        }
        return {
            code: "success",
            data: null as D,
            msg: "Successfully executed call action."
        }
    }
    
    async waitForFoldOption<D, E=Error>(): Response<D, E> {
        try {
            await this.page.waitForSelector(".game-decisions-ctn > .action-buttons > .fold", {timeout: this.default_timeout});
            const is_disabled = await this.page.$eval(".game-decisions-ctn > .action-buttons > .fold", (button: any) => button.disabled);
            if (is_disabled) {
                throw new Error("Fold option is disabled.")
            }
        } catch (err) {
            return {
                code: "error",
                error: new Error("No option to fold available.") as E
            }
        }
        return {
            code: "success",
            data: null as D,
            msg: "Successfully waited for fold option."
        }
    }

    async fold<D, E=Error>(): Response<D, E> {
        try {
            await this.page.waitForSelector(".game-decisions-ctn > .action-buttons > .fold", {timeout: this.default_timeout});
            await this.page.$eval(".game-decisions-ctn > .action-buttons > .fold", (button: any) => button.click());
        } catch (err) {
            return {
                code: "error",
                error: new Error("No option to fold available.") as E
            }
        }
        return {
            code: "success",
            data: null as D,
            msg: "Successfully executed fold action."
        }
    }
    
    async cancelUnnecessaryFold<D, E=Error>(): Response<D, E> {
        const fold_alert_text = "Are you sure that you want do an unnecessary fold?Do not show this again in this session? "
        try {
            await this.page.waitForSelector(".alert-1", {timeout: this.default_timeout});
            const text = await this.page.$eval(".alert-1 > .content", (div: any) => div.textContent);
            if (text === fold_alert_text) {
                await this.page.$eval(".alert-1 > .alert-1-buttons > .button-1.red", (button: any) => button.click());
            }
        } catch (err) {
            return {
                code: "error",
                error: new Error("No option to cancel unnecessary fold available.") as E
            }
        }
        return {
            code: "success",
            data: null as D,
            msg: "Successfully cancelled unnecessary fold."
        }
    }
    
    async waitForCheckOption<D, E=Error>(): Response<D, E> {
        try {
            await this.page.waitForSelector(".game-decisions-ctn > .action-buttons > .check", {timeout: this.default_timeout});
            const is_disabled = await this.page.$eval(".game-decisions-ctn > .action-buttons > .check", (button: any) => button.disabled);
            if (is_disabled) {
                throw new Error("Check option is disabled.")
            }
        } catch (err) {
            return {
                code: "error",
                error: new Error("No option to check available.") as E
            }
        }
        return {
            code: "success",
            data: null as D,
            msg: "Successfully waited for check option."
        }
    }
    
    async check<D, E=Error>(): Response<D, E> {
        try {
            await this.page.waitForSelector(".game-decisions-ctn > .action-buttons > .check", {timeout: this.default_timeout});
            await this.page.$eval(".game-decisions-ctn > .action-buttons > .check", (button: any) => button.click());
        } catch (err) {
            return {
                code: "error",
                error: new Error("No option to check available.") as E
            }
        }
        return {
            code: "success",
            data: null as D,
            msg: "Successfully executed check action."
        }
    }
    
    async waitForBetOption<D, E=Error>(): Response<D ,E> {
        try {
            await this.page.waitForSelector(".game-decisions-ctn > .action-buttons > .raise", {timeout: this.default_timeout});
            const is_disabled = await this.page.$eval(".game-decisions-ctn > .action-buttons > .raise", (button: any) => button.disabled);
            if (is_disabled) {
                throw new Error("Bet or raise option is disabled.")
            }
        } catch (err) {
            return {
                code: "error",
                error: new Error("No option to bet or raise available.") as E
            }
        }
        return {
            code: "success",
            data: null as D,
            msg: "Successfully waited for bet or raise option."
        }
    }
    
    async betOrRaise<D, E=Error>(bet_amount: number): Response<D, E> {
        try {
            const bet_action = await this.page.$eval(".game-decisions-ctn > .action-buttons > .raise", (button: any) => button.textContent);
            await this.page.$eval(".game-decisions-ctn > .action-buttons > .raise", (button: any) => button.click());
    
            if (bet_action === "Raise") {
                const res = await this.getCurrentBet();
                if (res.code === "success") {
                    const current_bet = res.data as number;
                    bet_amount += current_bet;
                }
            }
            await this.page.waitForSelector(".game-decisions-ctn > form > .raise-bet-value > div > input", {timeout: this.default_timeout});
            await this.page.focus(".game-decisions-ctn > form > .raise-bet-value > div > input");
            await sleep(this.default_timeout);
            await this.page.keyboard.type(bet_amount.toString(), {delay: 200});
            await this.page.waitForSelector(".game-decisions-ctn > form > .action-buttons > .bet", {timeout: this.default_timeout});
            await this.page.$eval(".game-decisions-ctn > form > .action-buttons > .bet", (input: any) => input.click());
        } catch (err) {
            return {
                code: "error",
                error: new Error(`Failed to bet with amount ${bet_amount}.`) as E
            }
        }
        return {
            code: "success",
            data: null as D,
            msg: `Successfully executed bet action with amount ${bet_amount}.`
        }
    }

    async getCurrentBet<D, E=Error>(): Response<D, E> {
        try {
            const el = await this.page.waitForSelector(".you-player > .table-player-bet-value", {timeout: this.default_timeout});
            const current_bet = await this.page.evaluate((el: any) => isNaN(el.textContent) ? '0' : el.textContent, el);
            return {
                code: "success",
                data: parseFloat(current_bet) as D,
                msg: `Successfully retrieved current bet amount: ${current_bet}`
            }
        } catch (err) {
            return {
                code: "error",
                error: new Error("No existing bet amount found.") as E
            }
        }
    }

    async waitForHandEnd<D, E=Error>(): Response<D, E> {
        try {
            await this.page.waitForSelector(".table-player.winner", {hidden: true, timeout: this.default_timeout * 10});
        } catch (err) {
            return {
                code: "error",
                error: new Error("Failed to wait for hand to finish.") as E
            }
        }
        return {
            code: "success",
            data: null as D,
            msg: "Waited for hand to finish."
        }
    }

    /**
     * Detect the dealer button seat number from the PokerNow DOM.
     *
     * PokerNow places a `.dealer-button-ctn` element inside the
     * table-player div that holds the button. The seat number is
     * extracted from the parent's class (e.g. `table-player-5` → seat 5).
     *
     * Returns the 1-based seat number, or null if undetectable.
     */
    async getDealerSeat<D, E=Error>(): Response<D, E> {
        try {
            const seatNum = await this.page.evaluate(() => {
                // Strategy 1: look for .dealer-button-ctn inside a table-player
                const dealerEl = (globalThis as any).document.querySelector(
                    ".dealer-button-ctn"
                );
                if (dealerEl) {
                    const parent = dealerEl.closest("[class*='table-player-']") as any;
                    if (parent) {
                        const match = parent.className.match(/table-player-(\d+)/);
                        if (match) return parseInt(match[1], 10);
                    }
                }

                // Strategy 2: look for the dealer class directly on a table-player
                const dealerPlayer = (globalThis as any).document.querySelector(
                    ".table-player.dealer, .table-player .dealer-button"
                );
                if (dealerPlayer) {
                    const el = dealerPlayer.closest("[class*='table-player-']") as any;
                    if (el) {
                        const match = el.className.match(/table-player-(\d+)/);
                        if (match) return parseInt(match[1], 10);
                    }
                }

                // Strategy 3: scan all table-players for a "D" badge or button icon
                const players = (globalThis as any).document.querySelectorAll(
                    "[class*='table-player-']"
                );
                for (const p of players) {
                    const text = p.innerText ?? "";
                    // PokerNow sometimes shows "Dealer" in the status line
                    if (/\bDealer\b/i.test(text) && !text.includes("Away")) {
                        const match = p.className.match(/table-player-(\d+)/);
                        if (match) return parseInt(match[1], 10);
                    }
                }

                return null;
            });

            if (seatNum !== null) {
                console.log(`[Puppeteer] Dealer button detected at seat ${seatNum}.`);
                return {
                    code: "success",
                    data: seatNum as D,
                    msg: `Dealer is at seat ${seatNum}.`
                };
            }

            return {
                code: "error",
                error: new Error("Dealer button not found in DOM.") as E
            };
        } catch (err) {
            return {
                code: "error",
                error: new Error("Failed to detect dealer seat.") as E
            };
        }
    }

    /**
     * Scrape all occupied seats and their player names from the DOM.
     * Returns an array of { seat: number, name: string } sorted by seat.
     */
    async getSeatedPlayers<D, E=Error>(): Response<D, E> {
        try {
            const players = await this.page.evaluate(() => {
                const result: { seat: number; name: string }[] = [];
                const divs = (globalThis as any).document.querySelectorAll(
                    ".table-player"
                );
                for (const div of divs) {
                    // Skip empty seats (those with a .table-player-seat class)
                    if (div.classList.contains("table-player-seat")) continue;
                    // Skip offline/standing players
                    const status = div.querySelector(".table-player-status-icon");
                    if (status) {
                        const statusText = status.textContent ?? "";
                        if (/standing.up|away/i.test(statusText)) continue;
                    }

                    const seatMatch = div.className.match(/table-player-(\d+)/);
                    const nameEl = div.querySelector(".table-player-name span");
                    if (seatMatch && nameEl) {
                        result.push({
                            seat: parseInt(seatMatch[1], 10),
                            name: (nameEl.textContent ?? "").trim(),
                        });
                    }
                }
                return result.sort((a: any, b: any) => a.seat - b.seat);
            });

            return {
                code: "success",
                data: players as D,
                msg: `Found ${(players as any[]).length} seated player(s).`
            };
        } catch (err) {
            return {
                code: "error",
                error: new Error("Failed to scrape seated players.") as E
            };
        }
    }

    /**
     * Scrape all player stacks from the DOM.
     * Returns an array of { seat, name, stack } for each seated player.
     * This provides the full table state needed by the AI agent integration.
     */
    async getAllPlayerStacks<D, E=Error>(): Response<D, E> {
        try {
            const players = await this.page.evaluate(() => {
                const result: { seat: number; name: string; stack: number }[] = [];
                const divs = (globalThis as any).document.querySelectorAll(
                    ".table-player"
                );
                for (const div of divs) {
                    // Skip empty seats
                    if (div.classList.contains("table-player-seat")) continue;
                    const status = div.querySelector(".table-player-status-icon");
                    if (status) {
                        const statusText = status.textContent ?? "";
                        if (/standing.up|away/i.test(statusText)) continue;
                    }

                    const seatMatch = div.className.match(/table-player-(\d+)/);
                    const nameEl = div.querySelector(".table-player-name span");
                    const stackEl = div.querySelector(".table-player-stack");

                    if (seatMatch && nameEl) {
                        const stackText = stackEl?.textContent?.replace(/[^0-9.]/g, "") ?? "0";
                        result.push({
                            seat: parseInt(seatMatch[1], 10),
                            name: (nameEl.textContent ?? "").trim(),
                            stack: parseFloat(stackText) || 0,
                        });
                    }
                }
                return result.sort((a: any, b: any) => a.seat - b.seat);
            });

            return {
                code: "success",
                data: players as D,
                msg: `Scraped stacks for ${(players as any[]).length} player(s).`
            };
        } catch (err) {
            return {
                code: "error",
                error: new Error("Failed to scrape player stacks.") as E
            };
        }
    }

    /**
     * Detect the bot's table position (BU, SB, BB, UTG, etc.)
     * by reading occupied seats and the dealer button from the DOM.
     *
     * Position labels match Table.getPositionLabels() so the whole
     * pipeline (engine, rlcard, LLM) sees the same strings.
     */
    async getMySeatPosition<D, E=Error>(myName: string): Response<D, E> {
        try {
            const position = await this.page.evaluate((name: string) => {
                const doc = (globalThis as any).document;
                const allPlayerDivs = [...doc.querySelectorAll('.table-player')] as any[];

                const getSeatNum = (div: any): number | null => {
                    const match = [...div.classList].find((c: string) =>
                        /^table-player-\d+$/.test(c)
                    );
                    return match ? parseInt(match.split('-').pop()!) : null;
                };

                // Occupied seats = those with a player name link
                const occupiedSeats = allPlayerDivs.filter((div: any) =>
                    div.querySelector('.table-player-name a') !== null
                );
                if (occupiedSeats.length === 0) return null;

                // Find hero's seat
                const mySeatDiv = occupiedSeats.find((div: any) => {
                    const nameLink = div.querySelector('.table-player-name a');
                    return nameLink && nameLink.textContent.trim() === name;
                });
                if (!mySeatDiv) return null;

                const mySeatNumber = getSeatNum(mySeatDiv);
                const occupiedSeatNumbers = occupiedSeats
                    .map(getSeatNum)
                    .filter((n: number | null): n is number => n !== null)
                    .sort((a: number, b: number) => a - b);

                const totalPlayers = occupiedSeatNumbers.length;
                if (totalPlayers === 0 || mySeatNumber === null) return null;

                // ── Detect dealer button seat ──
                let buttonSeatNumber: number | null = null;

                // Strategy 1: .dealer-button-ctn inside a table-player
                const dealerCtn = doc.querySelector('.dealer-button-ctn');
                if (dealerCtn) {
                    const parent = dealerCtn.closest("[class*='table-player-']");
                    if (parent) {
                        const m = parent.className.match(/table-player-(\d+)/);
                        if (m) buttonSeatNumber = parseInt(m[1], 10);
                    }
                }

                // Strategy 2: .dealer class on a table-player
                if (buttonSeatNumber === null) {
                    const dealerPlayer = doc.querySelector(
                        '.table-player.dealer, .table-player .dealer-button'
                    );
                    if (dealerPlayer) {
                        const el = dealerPlayer.closest("[class*='table-player-']");
                        if (el) {
                            const m = el.className.match(/table-player-(\d+)/);
                            if (m) buttonSeatNumber = parseInt(m[1], 10);
                        }
                    }
                }

                // Strategy 3: any element with "dealer" in its class
                if (buttonSeatNumber === null) {
                    const anyDealer = doc.querySelector('[class*="dealer"]');
                    if (anyDealer) {
                        const tp = anyDealer.closest('.table-player');
                        if (tp) {
                            const m = tp.className.match(/table-player-(\d+)/);
                            if (m) buttonSeatNumber = parseInt(m[1], 10);
                        }
                    }
                }

                // Last resort: first occupied seat
                if (buttonSeatNumber === null) {
                    buttonSeatNumber = occupiedSeatNumbers[0];
                }

                const buttonIndex = occupiedSeatNumbers.indexOf(buttonSeatNumber);
                if (buttonIndex === -1) return null;

                // Reorder seats clockwise starting from the button
                const orderedSeats: number[] = [];
                for (let i = 0; i < totalPlayers; i++) {
                    orderedSeats.push(
                        occupiedSeatNumbers[(buttonIndex + i) % totalPlayers]
                    );
                }

                // Position labels — must match Table.getPositionLabels()
                const LABELS: Record<number, string[]> = {
                    2:  ["SB", "BB"],
                    3:  ["BU", "SB", "BB"],
                    4:  ["BU", "SB", "BB", "UTG"],
                    5:  ["BU", "SB", "BB", "UTG", "CO"],
                    6:  ["BU", "SB", "BB", "UTG", "HJ", "CO"],
                    7:  ["BU", "SB", "BB", "UTG", "MP", "HJ", "CO"],
                    8:  ["BU", "SB", "BB", "UTG", "UTG+1", "MP", "HJ", "CO"],
                    9:  ["BU", "SB", "BB", "UTG", "UTG+1", "MP", "LJ", "HJ", "CO"],
                    10: ["BU", "SB", "BB", "UTG", "UTG+1", "MP", "MP", "LJ", "HJ", "CO"],
                };
                const positions = LABELS[totalPlayers] ?? LABELS[10]!;

                const myIndex = orderedSeats.indexOf(mySeatNumber);
                return myIndex >= 0 ? positions[myIndex % positions.length] : null;
            }, myName);

            if (position) {
                console.log(`[Puppeteer] Bot seat position: ${position}`);
                return {
                    code: "success",
                    data: position as D,
                    msg: `Bot is in position ${position}.`
                };
            }

            return {
                code: "error",
                error: new Error("Could not determine seat position.") as E
            };
        } catch (err) {
            return {
                code: "error",
                error: new Error("Failed to detect seat position.") as E
            };
        }
    }

    /**
     * Scrape ALL player positions from the DOM in one call.
     * Returns an array of { seat, name, position } sorted clockwise from dealer.
     * This is the single source of truth for position assignment.
     */
    async getAllPlayerPositions<D, E=Error>(): Response<D, E> {
        try {
            const result = await this.page.evaluate(() => {
                const doc = (globalThis as any).document;
                const allPlayerDivs = [...doc.querySelectorAll('.table-player')] as any[];

                const getSeatNum = (div: any): number | null => {
                    const match = [...div.classList].find((c: string) =>
                        /^table-player-\d+$/.test(c)
                    );
                    return match ? parseInt(match.split('-').pop()!) : null;
                };

                // Occupied seats = those with a player name
                const occupiedSeats = allPlayerDivs.filter((div: any) =>
                    div.querySelector('.table-player-name a') !== null
                );
                if (occupiedSeats.length === 0) return null;

                const seatData: { seat: number; name: string }[] = [];
                for (const div of occupiedSeats) {
                    const seat = getSeatNum(div);
                    const nameEl = div.querySelector('.table-player-name a');
                    const name = nameEl ? nameEl.textContent.trim() : "";
                    if (seat !== null && name) seatData.push({ seat, name });
                }
                seatData.sort((a: any, b: any) => a.seat - b.seat);

                const seatNumbers = seatData.map((s: any) => s.seat);
                const totalPlayers = seatNumbers.length;
                if (totalPlayers === 0) return null;

                // ── Detect dealer button seat ──
                let buttonSeat: number | null = null;

                const dealerCtn = doc.querySelector('.dealer-button-ctn');
                if (dealerCtn) {
                    const parent = dealerCtn.closest("[class*='table-player-']");
                    if (parent) {
                        const m = parent.className.match(/table-player-(\d+)/);
                        if (m) buttonSeat = parseInt(m[1], 10);
                    }
                }
                if (buttonSeat === null) {
                    const dp = doc.querySelector(
                        '.table-player.dealer, .table-player .dealer-button'
                    );
                    if (dp) {
                        const el = dp.closest("[class*='table-player-']");
                        if (el) {
                            const m = el.className.match(/table-player-(\d+)/);
                            if (m) buttonSeat = parseInt(m[1], 10);
                        }
                    }
                }
                if (buttonSeat === null) {
                    const anyDealer = doc.querySelector('[class*="dealer"]');
                    if (anyDealer) {
                        const tp = anyDealer.closest('.table-player');
                        if (tp) {
                            const m = tp.className.match(/table-player-(\d+)/);
                            if (m) buttonSeat = parseInt(m[1], 10);
                        }
                    }
                }
                if (buttonSeat === null) buttonSeat = seatNumbers[0];

                const btnIdx = seatNumbers.indexOf(buttonSeat);
                if (btnIdx === -1) return null;

                // Reorder clockwise from dealer
                const ordered: { seat: number; name: string }[] = [];
                for (let i = 0; i < totalPlayers; i++) {
                    ordered.push(seatData[(btnIdx + i) % totalPlayers]);
                }

                const LABELS: Record<number, string[]> = {
                    2:  ["SB", "BB"],
                    3:  ["BU", "SB", "BB"],
                    4:  ["BU", "SB", "BB", "UTG"],
                    5:  ["BU", "SB", "BB", "UTG", "CO"],
                    6:  ["BU", "SB", "BB", "UTG", "HJ", "CO"],
                    7:  ["BU", "SB", "BB", "UTG", "MP", "HJ", "CO"],
                    8:  ["BU", "SB", "BB", "UTG", "UTG+1", "MP", "HJ", "CO"],
                    9:  ["BU", "SB", "BB", "UTG", "UTG+1", "MP", "LJ", "HJ", "CO"],
                    10: ["BU", "SB", "BB", "UTG", "UTG+1", "MP", "MP", "LJ", "HJ", "CO"],
                };
                const positions = LABELS[totalPlayers] ?? LABELS[10]!;

                return ordered.map((p: any, i: number) => ({
                    seat: p.seat,
                    name: p.name,
                    position: positions[i % positions.length],
                }));
            });

            if (result && (result as any[]).length > 0) {
                console.log(
                    `[Puppeteer] All positions: ` +
                    (result as any[]).map((p: any) => `${p.name}=${p.position}`).join(", ")
                );
                return {
                    code: "success",
                    data: result as D,
                    msg: `Scraped positions for ${(result as any[]).length} player(s).`
                };
            }

            return {
                code: "error",
                error: new Error("Could not determine player positions.") as E
            };
        } catch (err) {
            return {
                code: "error",
                error: new Error("Failed to scrape player positions from DOM.") as E
            };
        }
    }
}