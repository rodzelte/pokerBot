"""
train_deep_cfr.py — Train Deep CFR agent for No-Limit Hold'em.

Deep CFR (Brown et al., 2019) uses neural networks to approximate the
cumulative advantage function (regret) and average strategy, allowing CFR
to scale to large games where tabular methods would need billions of entries.

This is the ONLY CFR variant that's truly viable for full NL Hold'em.

Architecture:
  - Outer iterations (T): each collects data and trains advantage networks
  - Inner traversals (K): external sampling tree walks per iteration
  - Final step: train strategy network on all collected strategy data

Usage:
  python train_deep_cfr.py                                    # defaults on NLHE
  python train_deep_cfr.py --iterations 500 --traversals 300  # larger training
  python train_deep_cfr.py --device cuda                      # GPU training

The trained model is saved to ./models/deep_cfr/ and loaded by the server:
  set RLCARD_MODEL_PATH=./models/deep_cfr/
  set RLCARD_AGENT_TYPE=deep_cfr
  python rlcard_server.py
"""
import argparse
import os
import logging
import signal
import time

import torch
import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import set_seed, tournament

from deep_cfr_agent import DeepCFRAgent
from strategy_table import StrategyTable

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("train_deep_cfr")


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    device = torch.device("cuda:0" if device_name == "cuda" else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return device


def main():
    parser = argparse.ArgumentParser(description="Train Deep CFR for NL Hold'em")
    parser.add_argument("--game", type=str, default="no-limit-holdem",
                        choices=["limit-holdem", "no-limit-holdem", "leduc-holdem"],
                        help="Game variant. Deep CFR shines on NL Hold'em.")
    parser.add_argument("--iterations", type=int, default=200,
                        help="Outer CFR iterations (each includes K traversals + network training)")
    parser.add_argument("--traversals", type=int, default=100,
                        help="Tree traversals per outer iteration (K in the paper)")
    parser.add_argument("--eval-every", type=int, default=20,
                        help="Evaluate and checkpoint every N outer iterations")
    parser.add_argument("--eval-num", type=int, default=1000,
                        help="Number of games per evaluation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="./models/deep_cfr")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate for advantage/strategy networks")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--train-steps", type=int, default=4000,
                        help="SGD steps per network training phase")
    parser.add_argument("--advantage-buffer-size", type=int, default=1_000_000,
                        help="Reservoir buffer capacity for advantage data")
    parser.add_argument("--strategy-buffer-size", type=int, default=1_000_000,
                        help="Reservoir buffer capacity for strategy data")
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    device_str = str(device)

    if device.type == "cuda":
        log.info(f"Device: {device} ({torch.cuda.get_device_name(device)})")
    else:
        log.info(f"Device: {device}")

    # Create environments
    env = rlcard.make(args.game, config={
        "game_num_players": 2,
        "seed": args.seed,
        "allow_step_back": True,
    })
    eval_env = rlcard.make(args.game, config={
        "game_num_players": 2,
        "seed": args.seed + 1,
    })

    # Create Deep CFR agent
    agent = DeepCFRAgent(
        env,
        model_path=args.save_dir,
        device=device_str,
        advantage_buffer_size=args.advantage_buffer_size,
        strategy_buffer_size=args.strategy_buffer_size,
        batch_size=args.batch_size,
        train_steps=args.train_steps,
        lr=args.lr,
    )

    # Resume from checkpoint
    resume_path = args.resume or args.save_dir
    agent.model_path = resume_path
    start_iteration = 1
    if agent.load():
        start_iteration = agent.iteration + 1
        log.info(f"Resumed Deep CFR from iteration {agent.iteration}")
        if start_iteration > args.iterations:
            log.info("Training already complete!")
            return
    else:
        log.info(f"Starting fresh Deep CFR training on {args.game}.")

    agent.model_path = args.save_dir

    random_agent = RandomAgent(num_actions=env.num_actions)
    eval_env.set_agents([agent, random_agent])

    os.makedirs(args.save_dir, exist_ok=True)

    log.info(
        f"Deep CFR config: {args.iterations} iterations x {args.traversals} traversals, "
        f"batch_size={args.batch_size}, train_steps={args.train_steps}, lr={args.lr}"
    )
    log.info(
        f"Buffers: advantage={args.advantage_buffer_size:,}, "
        f"strategy={args.strategy_buffer_size:,}"
    )
    log.info(f"Training iterations {start_iteration} to {args.iterations} on {args.game}...")

    # Graceful shutdown
    _shutdown = False
    _cur_iter = start_iteration

    def _handle_interrupt(signum, frame):
        nonlocal _shutdown
        if _shutdown:
            raise SystemExit(1)
        _shutdown = True
        log.info(f"\nInterrupt at iteration {_cur_iter}. Saving...")
        agent.save()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _handle_interrupt)

    perf_path = os.path.join(args.save_dir, "performance.csv")
    write_header = not os.path.exists(perf_path) or os.path.getsize(perf_path) == 0

    with open(perf_path, "a") as perf_file:
        if write_header:
            perf_file.write("iteration,reward,adv_loss,adv_buffer,strat_buffer\n")

        wall_start = time.perf_counter()
        window_start = wall_start

        for iteration in range(start_iteration, args.iterations + 1):
            _cur_iter = iteration

            # K traversals + advantage network training
            adv_loss = agent.train(num_traversals=args.traversals)

            if args.log_every > 0 and iteration % args.log_every == 0:
                elapsed = max(time.perf_counter() - window_start, 1e-9)
                rate = args.log_every / elapsed
                total_elapsed = time.perf_counter() - wall_start
                pct = iteration / args.iterations * 100
                remaining = (args.iterations - iteration) / max(rate, 0.01)
                adv_sizes = [len(b) for b in agent.advantage_buffers]
                strat_size = len(agent.strategy_buffer)
                log.info(
                    f"[DeepCFR] {iteration}/{args.iterations} ({pct:.1f}%) | "
                    f"Speed: {rate:.2f} iter/s | "
                    f"Adv loss: {adv_loss:.4f} | "
                    f"Buffers: adv={sum(adv_sizes):,} strat={strat_size:,} | "
                    f"Elapsed: {total_elapsed / 60:.1f}m | "
                    f"ETA: {remaining / 60:.1f}m"
                )
                window_start = time.perf_counter()

            if iteration % args.eval_every == 0:
                # Train a temporary strategy network for evaluation
                log.info(f"[DeepCFR] Training strategy network for eval at iteration {iteration}...")
                agent.train_strategy_network()

                eval_env.set_agents([agent, random_agent])
                reward = tournament(eval_env, args.eval_num)[0]
                pct = iteration / args.iterations * 100
                strat_size = len(agent.strategy_buffer)
                adv_sizes = [len(b) for b in agent.advantage_buffers]

                perf_file.write(
                    f"{iteration},{reward},{adv_loss:.6f},"
                    f"{sum(adv_sizes)},{strat_size}\n"
                )
                perf_file.flush()

                log.info(
                    f"[DeepCFR] Eval {iteration}/{args.iterations} ({pct:.1f}%) | "
                    f"Reward: {reward:.4f} | "
                    f"Strat buffer: {strat_size:,}"
                )
                agent.save()
                log.info(f"[DeepCFR] Checkpoint saved -> {args.save_dir}")

    # Final: train strategy network on all collected data
    log.info("[DeepCFR] Training final strategy network...")
    agent.train_strategy_network()
    agent.save()

    total = time.perf_counter() - wall_start
    log.info(f"Deep CFR training complete in {total / 60:.1f} min.")
    log.info(f"Model saved to {args.save_dir}")
    log.info(
        f"Strategy buffer: {len(agent.strategy_buffer):,} samples, "
        f"Advantage buffers: {[len(b) for b in agent.advantage_buffers]}"
    )
    log.info(f"\nTo serve: set RLCARD_AGENT_TYPE=deep_cfr")
    log.info(f"          set RLCARD_MODEL_PATH={args.save_dir}")
    log.info(f"          python rlcard_server.py")


if __name__ == "__main__":
    main()
