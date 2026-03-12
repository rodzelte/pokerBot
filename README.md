## PokerNow GPT

<a id="readme-top"></a>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
       <ul>
        <li><a href="#why-chatgpt-over-gto">Why ChatGPT over GTO?</a></li>
       </ul>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li>
      <a href="#supported-models">Supported Models</a>
    </li>
    <li>
      <a href="#rlcard-ai-python">RLCard AI (Python)</a>
      <ul>
        <li><a href="#training">Training</a></li>
        <li><a href="#unified-training-cli">Unified Training CLI</a></li>
        <li><a href="#self-play--opponent-pool">Self-Play & Opponent Pool</a></li>
        <li><a href="#exploration-noise-realistic-play">Exploration Noise (Realistic Play)</a></li>
        <li><a href="#stack-aware-rewards-bb-equity">Stack-Aware Rewards (BB Equity)</a></li>
        <li><a href="#running-the-rlcard-server">Running the Server</a></li>
        <li><a href="#decision-pipeline">Decision Pipeline</a></li>
      </ul>
    </li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

A Poker bot designed for [PokerNow](https://www.pokernow.club) using ChatGPT (or other models! check the Supported Models section below) to make decisions for the user. The bot web scrapes and fetches logs from PokerNow, building a model of the live game: the stakes, the user's hole cards, every player's position and stack size, the current pot size, the current street and shown community cards, and previous actions made by the the bot and other players. 

This model is used to formulate a query, fed into an LLM model. The output is parsed to reach a decision for the user, which is then executed automatically by the webdriver. The history of queries is maintained across a single hand and passed back into the model so that it can "remember" previous actions, such as who was the preflop aggressor.

During the operation of the bot, a cache is maintained to track the stats of every player in the table (VPIP, PFR). After the session ends, a SQLite database is updated with the players' stats, tracked by the player's name. This data will be retrieved the next time the user plays against that opponent again, building a stronger model of the opponent's tendencies the more the user plays against them. Each player's stats are used in the query, allowing the bot to make personalized exploitative adjustments to its strategy.

### Why ChatGPT (or other LLMs) over GTO?

A common question that is asked is why would someone use an LLM to determine the best possible actions over GTO strategy (Game Theory Optimal)? GTO strategies are generally considered a "perfect" way to play poker, with a balanced, aggressive, and unexploitable strategy.

GTO strategies are mostly solved for heads-up play (2 players) and has fewer data on how a hand should be played multi-handed (3+ players). As you increase the number of players participating in the pot, it becomes harder to follow an optimal strategy created for 2 or 3 players max. Weaker tables found online can have many loose players, increasing the chance of pots going multi-way postflop.

Furthermore, GTO strategies generally can't make use of opponents' VPIP (voluntary put money in pot) and PFR (pre-flop raise) stats to make exploitative adjustments to its strategy. One could argue that while GTO would perform the best against another bot using GTO or a human playing as close to GTO as possible, an agent utilizing an LLM could generate more profit against weaker opponents that stray far from GTO.

As ChatGPT and LLMs/generative models as a whole improve over time, we can and should expect ChatGPT to become a stronger, more profitable poker player.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

* [Node][Node-url]
* [Express][Express-url]
* [Puppeteer][Puppeteer-url]
* [SQLite][SQLite-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

### Installation

1. Get an Open AI API Key at [(https://platform.openai.com/docs/overview)](https://platform.openai.com/docs/overview)
2. Clone the repo
   ```sh
   git clone https://github.com/rodzelte/pokerBot
   ```
3. Install NPM packages
   ```sh
   npm install
   ```
4. Create a .env file in the base project directory ./pokernow-gpt and define your API Key
   ```js
   OPENAI_API_KEY = 'YOUR API KEY';
   GOOGLEAI_API_KEY = 'YOUR API KEY';
   ```
5. Update app/configs/ai-config.json to your desired provider and model_name (see supported models below for reference)
   ```json
   {
     "provider": "PROVIDER",
     "model_name": "MODEL"
   }
   ```
6. Run the app
   ```sh
   npx tsx app/index.ts
   ```
7. Follow the command line prompts to join an existing PokerNow game as a player. (ensure that the game is already setup with another player being the host, you can start a new game by going to https://www.pokernow.club/start-game)
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- SUPPORTED MODELS -->
## Supported Models
providers
---
"OpenAI", "Google"

models
---
OpenAI: "gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o"

Google: "gemini-1.5-flash", "gemini-1.0-pro", "gemini-1.5-pro"

<!-- RLCARD AI -->
## RLCard AI (Python)

The bot includes an RL-based poker AI powered by [rlcard](https://github.com/datamllab/rlcard). A Python FastAPI server wraps a trained agent (DQN, NFSP, or CFR) and serves decisions to the TypeScript bot over HTTP.

### Requirements

- Python 3.10+
- pip
- NVIDIA GPU and matching CUDA-enabled PyTorch build if you want GPU training

### Setup

1. Install Python dependencies
   ```sh
   cd python
   python -m pip install -r requirements.txt
   ```

2. *(Optional — for GPU training)* Install the CUDA build of PyTorch instead
   ```sh
   python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu124
   ```

3. Verify whether PyTorch can see your GPU
   ```sh
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
   ```

### Training

Three algorithms are available:

| Algorithm | Script | Best for |
|-----------|--------|----------|
| **DQN** | `train_dqn.py` | Fast training, good baseline, exploitative play |
| **NFSP** | `train_nfsp.py` | Converges toward Nash equilibrium, balanced play |
| **CFR** | `train_cfr.py` | Provably optimal in smaller games (Leduc, Limit) |

#### Train individual algorithms

```sh
cd python

# DQN — fast, exploitative
python train_dqn.py --episodes 500000 --device cuda

# NFSP — balanced Nash equilibrium
python train_nfsp.py --episodes 500000 --device cuda

# CFR — tabular, works best on smaller games
python train_cfr.py --iterations 10000 --game leduc-holdem
```

### Unified Training CLI

Train one or all algorithms with a single command using `train.py`:

```sh
# Train all three sequentially
python train.py all --episodes 1000000 --device cuda

# Train specific algorithms
python train.py dqn nfsp --episodes 500000 --device cuda

# Full training suite: self-play + noise + stack rewards
python train.py all --self-play --noise --stack-rewards --episodes 1000000 --device cuda
```

#### Common flags (DQN / NFSP)

| Flag | Default | Description |
|------|---------|-------------|
| `--episodes` | 50000 | Total training episodes |
| `--eval-every` | 10000 | Evaluate & checkpoint every N episodes |
| `--eval-num` | 1000 | Number of evaluation games per checkpoint |
| `--seed` | 42 | Random seed |
| `--save-dir` | `./models/<algo>` | Directory for checkpoints |
| `--resume` | *(auto)* | Path to checkpoint to resume from |
| `--device` | `auto` | `auto`, `cpu`, or `cuda` |
| `--log-every` | 10000 | Log training throughput every N episodes |

Checkpoints auto-resume — if you stop and restart, it picks up where it left off.

### Self-Play & Opponent Pool

Enable self-play to train the agent against past versions of itself instead of only random opponents. This is key to building human-competitive play.

```sh
# Self-play with opponent pool and curriculum
python train.py dqn --self-play --episodes 1000000 --device cuda

# Custom loss penalty (teaches risk aversion)
python train.py dqn --self-play --loss-multiplier 2.0 --episodes 500000
```

| Flag | Default | Description |
|------|---------|-------------|
| `--self-play` | off | Enable self-play with opponent pool |
| `--loss-multiplier` | 1.5 | Scale losses by this factor (risk aversion) |
| `--no-reward-shaping` | off | Disable all reward shaping |
| `--snapshot-interval` | auto | Episodes between opponent pool snapshots |

The curriculum gradually shifts opponents from easy to hard:
- **Warmup** (0-5%): 100% random — learn basic hand values
- **Early** (5-20%): 70% random, 30% self — start counter-strategies
- **Mid** (20-60%): 20% random, 50% self, 30% pool — diverse opponents
- **Late** (60-100%): 10% random, 40% self, 50% pool — mostly strong opponents

### Exploration Noise (Realistic Play)

Add poker-aware exploration noise that biases toward aggressive/creative actions instead of meaningless random folds. This produces human-like imperfect play and diverse training data.

```sh
# Enable noise for realistic play
python train.py dqn --noise --episodes 500000 --device cuda

# Customize noise parameters
python train.py nfsp --noise --noise-rate 0.1 --aggression-bias 0.7 --min-noise 0.02
```

| Flag | Default | Description |
|------|---------|-------------|
| `--noise` | off | Enable action-biased exploration |
| `--noise-rate` | 0.08 | Initial noise rate (decays over training) |
| `--aggression-bias` | 0.6 | When exploring, probability of aggressive action |
| `--min-noise` | 0.02 | Minimum noise rate at end of training |

Noise decays linearly from `--noise-rate` to `--min-noise` over training, so early episodes are creative and later episodes converge toward optimal.

### Stack-Aware Rewards (BB Equity)

Enable stack-aware reward shaping to teach the agent bankroll management. A virtual stack in big blinds (BB) carries across training hands — the more BBs the agent accumulates, the higher the reward.

```sh
# Enable stack-aware rewards
python train.py dqn --stack-rewards --episodes 500000 --device cuda

# Custom starting stack
python train.py nfsp --stack-rewards --initial-stack-bb 200 --device cuda

# Full suite: self-play + noise + stack equity
python train.py all --self-play --noise --stack-rewards --episodes 1000000 --device cuda
```

| Flag | Default | Description |
|------|---------|-------------|
| `--stack-rewards` | off | Enable BB-relative stack reward shaping |
| `--initial-stack-bb` | 100 | Starting stack in big blinds |

Stack-aware signals during training:
- **Big pot premium** — winning large pots earns extra reward (pot building)
- **Stack growth bonus** — growing above starting stack is rewarded
- **Deep stack bonus** — maintaining 80+ BB earns ongoing reward
- **Short stack urgency** — wins below 30 BB get extra reward (comeback play)
- **Stack protection** — extra loss penalty when stack is declining
- **Win streak momentum** — consecutive wins earn bonus

Evaluation also uses a stack-based tournament: the log shows profit in BB, win rate %, and peak stack reached.

### Running the RLCard Server

After training, start the server so the bot can query it:

```sh
cd python

# DQN (default)
python rlcard_server.py

# NFSP
set RLCARD_AGENT_TYPE=nfsp
set RLCARD_MODEL_PATH=./models/nfsp/checkpoint.pt
python rlcard_server.py

# CFR
set RLCARD_AGENT_TYPE=cfr
set RLCARD_MODEL_PATH=./models/cfr/cfr_model.pkl
python rlcard_server.py

# Custom device
set RLCARD_DEVICE=cuda
python rlcard_server.py
```

#### All flags at once (Recommended by developer)

```sh
python train.py all --self-play --noise --stack-rewards --episodes 1000000 --device cuda --loss-multiplier 2.0 --noise-rate 0.12 --aggression-bias 0.65 --min-noise 0.03 --initial-stack-bb 100 --rebuy-penalty 0.3 --eval-every 10000 --eval-num 1000 --log-every 10000 --seed 42
```

| Env Variable | Default | Description |
|-------------|---------|-------------|
| `RLCARD_AGENT_TYPE` | `dqn` | Agent type: `dqn`, `nfsp`, or `cfr` |
| `RLCARD_MODEL_PATH` | `./models/dqn/checkpoint.pt` | Path to trained model |
| `RLCARD_DEVICE` | `cpu` | Inference device |
| `RLCARD_PORT` | `5050` | Server port |

The server listens on `http://127.0.0.1:5050`. The bot automatically connects to it on startup — if the server is not running, the bot falls back to the local engine and LLM.

### Decision Pipeline

When the RLCard server is running, the bot uses this priority order:

1. **RLCard AI** (primary) — RL-trained DQN agent. High-confidence hands get larger bet sizing automatically.
2. **Local Poker Engine** (secondary) — Rule-based GTO engine, used when RLCard is unavailable.
3. **LLM API** (tertiary) — OpenAI/Google as a last-resort override when the engine wants to fold.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

zythonix.te@gmail.com

Project Link: [https://github.com/rodzelte/pokerBot](https://github.com/rodzelte/pokerBot)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
[Node-url]: https://nodejs.org/en
[Express-url]: https://expressjs.com/
[Puppeteer-url]: https://pptr.dev/
[SQLite-url]: https://www.sqlite.org/"# pokerBot" 
