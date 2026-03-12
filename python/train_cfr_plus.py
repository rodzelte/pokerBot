"""
train_cfr_plus.py — Train CFR+ agent for Hold'em.

CFR+ converges faster than vanilla CFR by flooring negative regrets to zero
and using linear strategy averaging. Typically reaches the same quality in
~10x fewer iterations.

Like vanilla CFR, this is tabular — best for smaller games (Limit, Leduc).
For No-Limit Hold'em, prefer Deep CFR (train_deep_cfr.py).

Usage:
  python train_cfr_plus.py                                        # 5k iterations on limit-holdem
  python train_cfr_plus.py --iterations 20000                     # more iterations
  python train_cfr_plus.py --game no-limit-holdem --iterations 2000  # NL (slow, large state space)

The trained model is saved to ./models/cfr_plus/ and loaded by the server:
  set RLCARD_MODEL_PATH=./models/cfr_plus/
  set RLCARD_AGENT_TYPE=cfr_plus
  python rlcard_server.py
"""
import argparse
import os
import logging
import signal
import time

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import set_seed, tournament

from cfr_variants import CFRPlusAgent
from strategy_table import StrategyTable

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("train_cfr_plus")


def main():
    parser = argparse.ArgumentParser(description="Train CFR+ agent for Hold'em")
    parser.add_argument("--game", type=str, default="limit-holdem",
                        choices=["limit-holdem", "no-limit-holdem", "leduc-holdem"])
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-num", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="./models/cfr_plus")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--log-every", type=int, default=500)
    args = parser.parse_args()

    set_seed(args.seed)

    if args.game == "no-limit-holdem":
        log.warning(
            "CFR+ on full NL Hold'em has an enormous state space. "
            "This will be very slow. Consider limit-holdem, leduc-holdem, or Deep CFR."
        )

    env = rlcard.make(args.game, config={
        "game_num_players": 2,
        "seed": args.seed,
        "allow_step_back": True,
    })
    eval_env = rlcard.make(args.game, config={
        "game_num_players": 2,
        "seed": args.seed + 1,
    })

    agent = CFRPlusAgent(env, model_path=args.save_dir)

    # Resume from checkpoint
    resume_path = args.resume or args.save_dir
    agent.model_path = resume_path
    start_iteration = 1
    if agent.load():
        start_iteration = agent.iteration + 1
        log.info(f"Resumed CFR+ from iteration {agent.iteration}")
        if start_iteration > args.iterations:
            log.info("Training already complete!")
            return
    else:
        log.info(f"Starting fresh CFR+ training on {args.game}.")

    agent.model_path = args.save_dir

    random_agent = RandomAgent(num_actions=env.num_actions)
    eval_env.set_agents([agent, random_agent])

    os.makedirs(args.save_dir, exist_ok=True)
    log.info(f"Training CFR+ iterations {start_iteration} to {args.iterations} on {args.game}...")

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
        # Export strategy table
        StrategyTable.export_from_cfr_agent(
            agent, "cfr_plus",
            os.path.join(args.save_dir, "cfr_plus_model.pkl"),
        )
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _handle_interrupt)

    perf_path = os.path.join(args.save_dir, "performance.csv")
    write_header = not os.path.exists(perf_path) or os.path.getsize(perf_path) == 0

    with open(perf_path, "a") as perf_file:
        if write_header:
            perf_file.write("iteration,reward,info_sets\n")

        wall_start = time.perf_counter()
        window_start = wall_start

        for iteration in range(start_iteration, args.iterations + 1):
            _cur_iter = iteration
            agent.train()

            if args.log_every > 0 and iteration % args.log_every == 0:
                elapsed = max(time.perf_counter() - window_start, 1e-9)
                rate = args.log_every / elapsed
                total_elapsed = time.perf_counter() - wall_start
                pct = iteration / args.iterations * 100
                remaining = (args.iterations - iteration) / max(rate, 0.01)
                info_sets = len(agent.regrets)
                log.info(
                    f"[CFR+] {iteration}/{args.iterations} ({pct:.1f}%) | "
                    f"Speed: {rate:.1f} iter/s | "
                    f"Info sets: {info_sets:,} | "
                    f"Elapsed: {total_elapsed / 60:.1f}m | "
                    f"ETA: {remaining / 60:.1f}m"
                )
                window_start = time.perf_counter()

            if iteration % args.eval_every == 0:
                eval_env.set_agents([agent, random_agent])
                reward = tournament(eval_env, args.eval_num)[0]
                info_sets = len(agent.regrets)
                pct = iteration / args.iterations * 100

                perf_file.write(f"{iteration},{reward},{info_sets}\n")
                perf_file.flush()

                log.info(
                    f"[CFR+] Eval {iteration}/{args.iterations} ({pct:.1f}%) | "
                    f"Reward: {reward:.4f} | Info sets: {info_sets:,}"
                )
                agent.save()

    # Final save + export strategy table
    agent.save()
    StrategyTable.export_from_cfr_agent(
        agent, "cfr_plus",
        os.path.join(args.save_dir, "cfr_plus_model.pkl"),
    )

    total = time.perf_counter() - wall_start
    log.info(f"CFR+ training complete in {total / 60:.1f} min.")
    log.info(f"Model saved to {args.save_dir}")
    log.info(f"Info sets explored: {len(agent.regrets):,}")


if __name__ == "__main__":
    main()
