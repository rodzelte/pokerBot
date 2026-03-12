"""
train_mccfr.py — Train External Sampling Monte Carlo CFR for Hold'em.

MCCFR samples opponent actions instead of traversing all of them,
dramatically reducing per-iteration cost. Each iteration is cheaper but
noisier — you trade precision for speed.

Best for: medium-sized games where vanilla CFR/CFR+ is too slow per iteration
but Deep CFR is overkill. Also useful on Limit Hold'em to get more iterations
in the same wall-clock time.

Usage:
  python train_mccfr.py                                         # 10k iterations on limit-holdem
  python train_mccfr.py --iterations 50000                      # more iterations
  python train_mccfr.py --game no-limit-holdem --iterations 5000  # NL (tabular, still large)
  python train_mccfr.py --no-plus                                # vanilla regret (no floor)

The trained model is saved to ./models/mccfr/ and loaded by the server:
  set RLCARD_MODEL_PATH=./models/mccfr/
  set RLCARD_AGENT_TYPE=mccfr
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

from cfr_variants import MCCFRAgent
from strategy_table import StrategyTable

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("train_mccfr")


def main():
    parser = argparse.ArgumentParser(description="Train MCCFR agent for Hold'em")
    parser.add_argument("--game", type=str, default="limit-holdem",
                        choices=["limit-holdem", "no-limit-holdem", "leduc-holdem"])
    parser.add_argument("--iterations", type=int, default=10000,
                        help="MCCFR iterations (each is fast; use more than vanilla CFR)")
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--eval-num", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="./models/mccfr")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--no-plus", action="store_true",
                        help="Disable CFR+ regret floor (use vanilla regret matching)")
    args = parser.parse_args()

    set_seed(args.seed)

    if args.game == "no-limit-holdem":
        log.warning(
            "MCCFR on full NL Hold'em is feasible but the tabular state space "
            "is still enormous. For NL, Deep CFR is recommended. "
            "Continuing with tabular MCCFR..."
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

    use_plus = not args.no_plus
    agent = MCCFRAgent(env, model_path=args.save_dir, use_plus=use_plus)
    variant = "MCCFR+" if use_plus else "MCCFR"

    # Resume
    resume_path = args.resume or args.save_dir
    agent.model_path = resume_path
    start_iteration = 1
    if agent.load():
        start_iteration = agent.iteration + 1
        log.info(f"Resumed {variant} from iteration {agent.iteration}")
        if start_iteration > args.iterations:
            log.info("Training already complete!")
            return
    else:
        log.info(f"Starting fresh {variant} training on {args.game}.")

    agent.model_path = args.save_dir

    random_agent = RandomAgent(num_actions=env.num_actions)
    eval_env.set_agents([agent, random_agent])

    os.makedirs(args.save_dir, exist_ok=True)
    log.info(f"Training {variant} iterations {start_iteration} to {args.iterations} on {args.game}...")

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
        StrategyTable.export_from_cfr_agent(
            agent, "mccfr",
            os.path.join(args.save_dir, "mccfr_model.pkl"),
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
                    f"[{variant}] {iteration}/{args.iterations} ({pct:.1f}%) | "
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
                    f"[{variant}] Eval {iteration}/{args.iterations} ({pct:.1f}%) | "
                    f"Reward: {reward:.4f} | Info sets: {info_sets:,}"
                )
                agent.save()

    # Final save + export
    agent.save()
    StrategyTable.export_from_cfr_agent(
        agent, "mccfr",
        os.path.join(args.save_dir, "mccfr_model.pkl"),
    )

    total = time.perf_counter() - wall_start
    log.info(f"{variant} training complete in {total / 60:.1f} min.")
    log.info(f"Model saved to {args.save_dir}")
    log.info(f"Info sets explored: {len(agent.regrets):,}")


if __name__ == "__main__":
    main()
