"""
train_nfsp.py — Train rlcard's NFSP (Neural Fictitious Self-Play) agent for No-Limit Hold'em.

NFSP learns by combining reinforcement learning with supervised learning on its
own average strategy, converging towards a Nash equilibrium in imperfect-information games.

Usage:
  python train_nfsp.py                          # 50k episodes, saves to ./models/nfsp/
  python train_nfsp.py --episodes 200000        # more episodes for better play
  python train_nfsp.py --resume ./models/nfsp   # resume from checkpoint
  python train_nfsp.py --self-play              # enable self-play + opponent pool

The trained model is saved to ./models/nfsp/ and can be loaded by the server:
  set RLCARD_MODEL_PATH=./models/nfsp/checkpoint.pt
  set RLCARD_AGENT_TYPE=nfsp
  python rlcard_server.py
"""
import argparse
import os
import logging
import signal
import time

import torch
import rlcard
from rlcard.agents import NFSPAgent, RandomAgent
from rlcard.utils import (
    set_seed,
    tournament,
    reorganize,
)

from opponents import (
    OpponentPool,
    CurriculumScheduler,
    RewardShaper,
    StackTracker,
    NoisyActionWrapper,
    get_snapshot_interval,
    stack_tournament,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("train_nfsp")


def resolve_device(device_name: str) -> torch.device:
    """Resolve the requested torch device for training."""
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but no CUDA-capable GPU is available to PyTorch. "
            "Install a CUDA-enabled torch build and verify your NVIDIA driver."
        )

    device = torch.device("cuda:0" if device_name == "cuda" else "cpu")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    return device


def log_device_info(device: torch.device) -> None:
    """Log the selected training device and relevant runtime details."""
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        log.info(f"Device: {device} ({gpu_name})")
    else:
        log.info(f"Device: {device}")
    log.info(f"Torch threads: intraop={torch.get_num_threads()}")


def save_checkpoint(agent: NFSPAgent, checkpoint_path: str, episode: int, agent_config: dict) -> None:
    """Persist model weights and training metadata for NFSP."""
    torch.save({
        # RL (best response) network
        "q_net": agent._rl_agent.q_estimator.qnet.state_dict(),
        "target_net": agent._rl_agent.target_estimator.qnet.state_dict(),
        # Average policy (supervised learning) network
        "avg_net": agent.policy_network.state_dict(),
        "total_t": agent._rl_agent.total_t,
        "episode": episode,
        "agent_type": "nfsp",
        "agent_config": agent_config,
    }, checkpoint_path)


def main():
    parser = argparse.ArgumentParser(description="Train rlcard NFSP agent for NL Hold'em")
    parser.add_argument("--episodes", type=int, default=50000, help="Number of training episodes")
    parser.add_argument("--eval-every", type=int, default=10000, help="Evaluate every N episodes")
    parser.add_argument("--eval-num", type=int, default=1000, help="Number of evaluation games")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="./models/nfsp")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    parser.add_argument("--rl-lr", type=float, default=0.0005, help="RL (best response) learning rate")
    parser.add_argument("--sl-lr", type=float, default=0.001, help="Supervised learning (avg policy) learning rate")
    parser.add_argument("--anticipatory-param", type=float, default=0.1,
                        help="Anticipatory parameter η: probability of using best-response policy. "
                             "Lower values -> more average strategy (closer to Nash). Range [0, 1].")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="Training device. 'auto' uses CUDA when available.")
    parser.add_argument("--cpu-threads", type=int, default=0,
                        help="Override PyTorch CPU thread count (0 keeps the default).")
    parser.add_argument("--log-every", type=int, default=10000,
                        help="Log throughput every N episodes during long training runs.")
    parser.add_argument("--self-play", action="store_true",
                        help="Enable self-play with opponent pool and curriculum training.")
    parser.add_argument("--loss-multiplier", type=float, default=1.5,
                        help="Reward shaping: scale losses by this factor (teaches risk aversion).")
    parser.add_argument("--no-reward-shaping", action="store_true",
                        help="Disable reward shaping (use raw payoffs only).")
    parser.add_argument("--snapshot-interval", type=int, default=0,
                        help="Episodes between opponent pool snapshots (0 = auto).")
    parser.add_argument("--noise", action="store_true",
                        help="Enable action-biased exploration for realistic imperfect play.")
    parser.add_argument("--noise-rate", type=float, default=0.08,
                        help="Initial noise rate (decays to --min-noise over training).")
    parser.add_argument("--aggression-bias", type=float, default=0.6,
                        help="When exploring, probability of aggressive action vs passive.")
    parser.add_argument("--min-noise", type=float, default=0.02,
                        help="Minimum noise rate at end of training.")
    parser.add_argument("--stack-rewards", action="store_true",
                        help="Enable stack-aware BB-relative reward shaping (equity/bankroll).")
    parser.add_argument("--initial-stack-bb", type=float, default=100.0,
                        help="Initial stack in big blinds for stack tracking.")
    parser.add_argument("--rebuy-penalty", type=float, default=0.15,
                        help="Reward penalty per rebuy (escalates: N * this value). Higher = harsher.")
    args = parser.parse_args()

    set_seed(args.seed)
    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
    device = resolve_device(args.device)
    log_device_info(device)

    # Create environment
    env = rlcard.make("no-limit-holdem", config={
        "game_num_players": 2,
        "seed": args.seed,
    })
    eval_env = rlcard.make("no-limit-holdem", config={
        "game_num_players": 2,
        "seed": args.seed + 1,
    })

    # NFSP agent config
    agent_config = dict(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        hidden_layers_sizes=[256, 256, 128],
        q_mlp_layers=[256, 256, 128],
        device=device,
        anticipatory_param=args.anticipatory_param,
        rl_learning_rate=args.rl_lr,
        sl_learning_rate=args.sl_lr,
        reservoir_buffer_capacity=100000,
        q_replay_memory_size=50000,
        batch_size=64,
        min_buffer_size_to_learn=256,
        evaluate_with="average_policy",
    )

    # Create NFSP agent
    checkpoint_path = os.path.join(args.save_dir, "checkpoint.pt")
    resume_path = args.resume or args.save_dir
    resume_ckpt = os.path.join(resume_path, "checkpoint.pt")
    start_episode = 1

    agent = NFSPAgent(**agent_config)

    if os.path.exists(resume_ckpt):
        log.info(f"Found checkpoint: {resume_ckpt} — resuming...")
        ckpt = torch.load(resume_ckpt, map_location=device, weights_only=False)

        # Load RL network
        agent._rl_agent.q_estimator.qnet.load_state_dict(ckpt["q_net"])
        agent._rl_agent.target_estimator.qnet.load_state_dict(ckpt["target_net"])
        # Load average policy network
        if "avg_net" in ckpt:
            agent.policy_network.load_state_dict(ckpt["avg_net"])

        agent._rl_agent.total_t = 0  # Reset for buffer warm-up

        saved_episode = ckpt.get("episode", 0)
        if saved_episode == 0:
            perf_csv = os.path.join(args.save_dir, "performance.csv")
            if os.path.exists(perf_csv):
                with open(perf_csv) as f:
                    lines = [l.strip() for l in f if l.strip() and not l.startswith("episode")]
                if lines:
                    saved_episode = int(lines[-1].split(",")[0])
                    log.info(f"Recovered episode count from performance.csv: {saved_episode}")
        start_episode = saved_episode + 1
        log.info(f"Resumed at episode {saved_episode} (total_t reset for replay warm-up)")
        if start_episode > args.episodes:
            log.info("Training already complete!")
            return
    else:
        log.info("No checkpoint found — starting fresh NFSP training.")

    # Random opponent for training
    random_agent = RandomAgent(num_actions=env.num_actions)

    # Self-play setup
    opponent_pool = None
    curriculum = None
    reward_shaper = RewardShaper(
        loss_multiplier=args.loss_multiplier,
        rebuy_penalty=args.rebuy_penalty,
        enabled=not args.no_reward_shaping,
    )

    if args.self_play:
        snap_interval = args.snapshot_interval or get_snapshot_interval(args.episodes)
        opponent_pool = OpponentPool(
            num_actions=env.num_actions,
            device=device,
        )
        curriculum = CurriculumScheduler(args.episodes)
        log.info(f"Self-play ENABLED — snapshot every {snap_interval} episodes")
        log.info(f"Reward shaping: loss_multiplier={args.loss_multiplier}, "
                 f"enabled={not args.no_reward_shaping}")
    else:
        snap_interval = 0
        log.info("Self-play DISABLED — training against random opponent")

    # Stack tracking for BB-relative rewards
    stack_tracker = None
    if args.stack_rewards:
        stack_tracker = StackTracker(initial_stack_bb=args.initial_stack_bb,
                                     rebuy_penalty=args.rebuy_penalty)
        log.info(f"Stack-aware rewards ENABLED — initial stack: {args.initial_stack_bb} BB, "
                 f"rebuy penalty: {args.rebuy_penalty}")

    # Action noise for realistic imperfect play
    train_hero = agent
    noisy_agent = None
    if args.noise:
        noisy_agent = NoisyActionWrapper(
            agent,
            noise_rate=args.noise_rate,
            aggression_bias=args.aggression_bias,
            min_noise=args.min_noise,
            total_episodes=args.episodes,
        )
        train_hero = noisy_agent
        log.info(f"Exploration noise ENABLED — rate: {args.noise_rate}, "
                 f"aggression: {args.aggression_bias}, decay to: {args.min_noise}")

    env.set_agents([train_hero, random_agent])
    eval_env.set_agents([agent, random_agent])  # eval always uses pure agent

    os.makedirs(args.save_dir, exist_ok=True)

    log.info(f"Training NFSP episodes {start_episode} to {args.episodes}...")
    log.info(f"Anticipatory param η={args.anticipatory_param} "
             f"(lower -> closer to Nash equilibrium)")

    # Graceful shutdown on Ctrl+C
    _shutdown_requested = False
    _current_episode = start_episode

    def _handle_interrupt(signum, frame):
        nonlocal _shutdown_requested
        if _shutdown_requested:
            log.warning("Second interrupt — force exiting without save.")
            raise SystemExit(1)
        _shutdown_requested = True
        log.info(f"\nInterrupt received at episode {_current_episode}. Saving checkpoint...")
        save_checkpoint(agent, checkpoint_path, _current_episode, agent_config)
        log.info(f"Checkpoint saved to {checkpoint_path} (episode {_current_episode})")
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _handle_interrupt)

    # Performance log
    perf_path = os.path.join(args.save_dir, "performance.csv")
    write_header = not os.path.exists(perf_path) or os.path.getsize(perf_path) == 0

    with open(perf_path, "a") as perf_file:
        if write_header:
            perf_file.write("episode,reward\n")

        wall_start = time.perf_counter()
        window_start = wall_start

        for episode in range(start_episode, args.episodes + 1):
            _current_episode = episode

            # Decay exploration noise
            if noisy_agent:
                noisy_agent.update_noise(episode)

            # Select opponent for this episode
            opp_name = "random"
            if opponent_pool and curriculum:
                weights = curriculum.get_opponent_weights(episode)
                opponent, opp_name = opponent_pool.sample(weights)
                env.set_agents([train_hero, opponent])

            # Generate training data
            trajectories, payoffs = env.run(is_training=True)

            # Update stack tracker and get metrics for reward shaping
            stack_metrics = None
            if stack_tracker:
                stack_metrics = stack_tracker.update(payoffs[0])

            # Apply reward shaping (with stack-aware signals if enabled)
            shaped_payoffs = reward_shaper.shape(payoffs, trajectories, stack_metrics=stack_metrics)

            trajectories = reorganize(trajectories, shaped_payoffs)

            for ts in trajectories[0]:
                agent.feed(ts)

            # Snapshot agent to opponent pool on schedule
            if opponent_pool and snap_interval and episode % snap_interval == 0:
                if curriculum and curriculum.should_snapshot(episode, snap_interval):
                    opponent_pool.add_snapshot(agent, name=f"snap-{episode}")

            # Manual save trigger
            trigger_path = os.path.join(args.save_dir, "SAVE")
            if os.path.exists(trigger_path):
                os.remove(trigger_path)
                save_checkpoint(agent, checkpoint_path, episode, agent_config)
                log.info(f"Manual save triggered at episode {episode} — checkpoint saved.")

            if args.log_every > 0 and episode % args.log_every == 0:
                elapsed = max(time.perf_counter() - window_start, 1e-9)
                eps_per_sec = args.log_every / elapsed
                total_elapsed = max(time.perf_counter() - wall_start, 1e-9)
                pct = episode / args.episodes * 100
                remaining = (args.episodes - episode) / max(eps_per_sec, 0.01)
                log.info(
                    f"[NFSP] {episode}/{args.episodes} ({pct:.1f}%) | "
                    f"Speed: {eps_per_sec:.1f} eps/s | "
                    f"Elapsed: {total_elapsed / 60:.1f} min | "
                    f"ETA: {remaining / 60:.1f} min"
                )
                window_start = time.perf_counter()

            # Periodic evaluation + checkpoint
            if episode % args.eval_every == 0:
                # Always eval against random for consistent baseline
                eval_env.set_agents([agent, random_agent])
                pct = episode / args.episodes * 100
                phase = curriculum.get_phase(episode) if curriculum else "random"
                pool_size = opponent_pool.size if opponent_pool else 1
                noise_info = f" Noise:{noisy_agent.noise_rate:.3f}" if noisy_agent else ""

                if args.stack_rewards:
                    seval = stack_tournament(eval_env, args.eval_num, args.initial_stack_bb)
                    reward = seval["avg_payoff"]
                    rebuy_info = f" Rebuys:{seval['rebuys']}" if seval['rebuys'] > 0 else ""
                    stack_info = (f" | +{seval['profit_bb']:.1f}BB"
                                 f" WR:{seval['win_rate_pct']:.0f}%"
                                 f" Peak:{seval['peak_bb']:.0f}BB{rebuy_info}")
                    perf_file.write(f"{episode},{reward},{seval['profit_bb']:.2f},{seval['win_rate_pct']:.1f},{seval['rebuys']}\n")
                else:
                    reward = tournament(eval_env, args.eval_num)[0]
                    stack_info = ""
                    perf_file.write(f"{episode},{reward}\n")
                perf_file.flush()

                log.info(
                    f"[NFSP] Checkpoint {episode}/{args.episodes} ({pct:.1f}%) | "
                    f"Eval reward: {reward:.4f}{stack_info} | "
                    f"Phase: {phase} | Pool: {pool_size}{noise_info}"
                )
                if stack_tracker:
                    log.info(f"[NFSP] Training bankroll: {stack_tracker.summary()}")
                save_checkpoint(agent, checkpoint_path, episode, agent_config)
                log.info(f"[NFSP] Saved -> {checkpoint_path}")

        if args.episodes % args.eval_every != 0:
            save_checkpoint(agent, checkpoint_path, args.episodes, agent_config)
            log.info(f"Final checkpoint saved to {checkpoint_path} (episode {args.episodes})")

    log.info("NFSP Training complete!")
    log.info(f"Model saved to {checkpoint_path}")
    log.info(f"To use: set RLCARD_MODEL_PATH={checkpoint_path} & set RLCARD_AGENT_TYPE=nfsp")


if __name__ == "__main__":
    main()
