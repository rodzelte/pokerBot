"""
train.py — Unified training entry point for all rlcard algorithms.

Train DQN, NFSP, and/or CFR agents with a single command.

Usage:
  python train.py dqn                      # train DQN only
  python train.py nfsp                     # train NFSP only
  python train.py cfr                      # train CFR only
  python train.py all                      # train all three sequentially
  python train.py dqn nfsp                 # train DQN then NFSP
  python train.py all --episodes 100000    # all three with custom episode count

Algorithm-specific flags are forwarded automatically.  Pass --help after an
algorithm name to see its specific options:
  python train.py dqn --help

Examples:
  python train.py all --episodes 100000 --device cuda
  python train.py dqn nfsp --episodes 50000 --seed 123
  python train.py cfr --iterations 10000 --game leduc-holdem
  python train.py all --episodes 200000 --eval-every 10000

After training, start the server with the desired agent:
  set RLCARD_AGENT_TYPE=nfsp
  set RLCARD_MODEL_PATH=./models/nfsp/checkpoint.pt
  python rlcard_server.py
"""
import argparse
import subprocess
import sys
import os
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("train")

ALGORITHMS = ["dqn", "nfsp", "cfr", "cfr_plus", "mccfr", "deep_cfr"]

ALGO_SCRIPTS = {
    "dqn": "train_dqn.py",
    "nfsp": "train_nfsp.py",
    "cfr": "train_cfr.py",
    "cfr_plus": "train_cfr_plus.py",
    "mccfr": "train_mccfr.py",
    "deep_cfr": "train_deep_cfr.py",
}

# Flags shared across DQN/NFSP (episode-based training)
SHARED_EPISODE_FLAGS = ["--episodes", "--eval-every", "--eval-num", "--seed",
                        "--device", "--cpu-threads", "--log-every"]

# Flags shared across all algorithms
SHARED_FLAGS = ["--seed", "--log-every"]


def build_args(algo: str, parsed: argparse.Namespace, extra: list[str]) -> list[str]:
    """Build the argument list to forward to an algorithm-specific training script."""
    args = []

    # Forward shared flags that apply
    if algo in ("dqn", "nfsp"):
        if parsed.episodes is not None:
            args += ["--episodes", str(parsed.episodes)]
        if parsed.device is not None:
            args += ["--device", parsed.device]
        if parsed.cpu_threads is not None:
            args += ["--cpu-threads", str(parsed.cpu_threads)]
        # Self-play flags (DQN/NFSP only)
        if parsed.self_play:
            args += ["--self-play"]
        if parsed.loss_multiplier is not None:
            args += ["--loss-multiplier", str(parsed.loss_multiplier)]
        if parsed.no_reward_shaping:
            args += ["--no-reward-shaping"]
        if parsed.snapshot_interval is not None:
            args += ["--snapshot-interval", str(parsed.snapshot_interval)]
        # Noise / realistic play flags
        if parsed.noise:
            args += ["--noise"]
        if parsed.noise_rate is not None:
            args += ["--noise-rate", str(parsed.noise_rate)]
        if parsed.aggression_bias is not None:
            args += ["--aggression-bias", str(parsed.aggression_bias)]
        if parsed.min_noise is not None:
            args += ["--min-noise", str(parsed.min_noise)]
        # Stack-aware reward flags
        if parsed.stack_rewards:
            args += ["--stack-rewards"]
        if parsed.initial_stack_bb is not None:
            args += ["--initial-stack-bb", str(parsed.initial_stack_bb)]
        if parsed.rebuy_penalty is not None:
            args += ["--rebuy-penalty", str(parsed.rebuy_penalty)]
    elif algo in ("cfr", "cfr_plus", "mccfr"):
        if parsed.iterations is not None:
            args += ["--iterations", str(parsed.iterations)]
        if parsed.cfr_game is not None:
            args += ["--game", parsed.cfr_game]
        if algo == "mccfr" and parsed.mccfr_no_plus:
            args += ["--no-plus"]
    elif algo == "deep_cfr":
        if parsed.deep_cfr_iterations is not None:
            args += ["--iterations", str(parsed.deep_cfr_iterations)]
        elif parsed.iterations is not None:
            args += ["--iterations", str(parsed.iterations)]
        if parsed.traversals is not None:
            args += ["--traversals", str(parsed.traversals)]
        if parsed.device is not None:
            args += ["--device", parsed.device]
        if parsed.deep_cfr_lr is not None:
            args += ["--lr", str(parsed.deep_cfr_lr)]
        if parsed.deep_cfr_batch_size is not None:
            args += ["--batch-size", str(parsed.deep_cfr_batch_size)]
        if parsed.deep_cfr_train_steps is not None:
            args += ["--train-steps", str(parsed.deep_cfr_train_steps)]
        if parsed.cfr_game is not None:
            args += ["--game", parsed.cfr_game]

    # Common flags
    if parsed.eval_every is not None:
        args += ["--eval-every", str(parsed.eval_every)]
    if parsed.eval_num is not None:
        args += ["--eval-num", str(parsed.eval_num)]
    if parsed.seed is not None:
        args += ["--seed", str(parsed.seed)]
    if parsed.log_every is not None:
        args += ["--log-every", str(parsed.log_every)]
    if parsed.save_dir is not None:
        args += ["--save-dir", parsed.save_dir]
    else:
        args += ["--save-dir", f"./models/{algo}"]

    # Forward any unrecognised flags verbatim
    args += extra

    return args


def run_training(algo: str, args: list[str]) -> bool:
    """Run a training script as a subprocess. Returns True on success."""
    script = ALGO_SCRIPTS[algo]
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script)

    if not os.path.exists(script_path):
        log.error(f"Training script not found: {script_path}")
        return False

    cmd = [sys.executable, script_path] + args
    log.info(f"{'=' * 60}")
    log.info(f"Starting {algo.upper()} training")
    log.info(f"Command: {' '.join(cmd)}")
    log.info(f"{'=' * 60}")

    start = time.perf_counter()
    result = subprocess.run(cmd)
    elapsed = time.perf_counter() - start

    if result.returncode == 0:
        log.info(f"{algo.upper()} training completed in {elapsed / 60:.1f} min")
        return True
    else:
        log.error(f"{algo.upper()} training failed (exit code {result.returncode}) "
                  f"after {elapsed / 60:.1f} min")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Unified training for rlcard poker agents (DQN, NFSP, CFR)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py all                                      # all three with defaults
  python train.py dqn nfsp --episodes 100000               # train DQN and NFSP
  python train.py cfr --iterations 20000                   # train CFR only
  python train.py all --device cuda                        # train all on GPU
  python train.py all --self-play --episodes 500000        # self-play (beats humans!)
  python train.py dqn --self-play --loss-multiplier 2.0    # aggressive loss penalty
  python train.py all --noise --stack-rewards              # realistic play + BB equity
  python train.py dqn --noise --stack-rewards --self-play  # full training suite
  python train.py cfr_plus mccfr --iterations 10000        # CFR variants
  python train.py deep_cfr --device cuda --traversals 300  # Deep CFR on GPU
        """,
    )

    parser.add_argument(
        "algorithms", nargs="+",
        choices=ALGORITHMS + ["all"],
        help="Which algorithm(s) to train. Use 'all' for DQN + NFSP + CFR.",
    )

    # Shared options
    parser.add_argument("--episodes", type=int, default=None,
                        help="Training episodes for DQN/NFSP (default: 50000)")
    parser.add_argument("--iterations", type=int, default=None,
                        help="Training iterations for CFR (default: 5000)")
    parser.add_argument("--eval-every", type=int, default=None,
                        help="Evaluate every N episodes/iterations")
    parser.add_argument("--eval-num", type=int, default=None,
                        help="Number of evaluation games")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None,
                        help="Training device for DQN/NFSP")
    parser.add_argument("--cpu-threads", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Override save directory (default: ./models/<algo>)")
    parser.add_argument("--cfr-game", type=str, default=None,
                        choices=["limit-holdem", "no-limit-holdem", "leduc-holdem"],
                        help="Game variant for CFR/CFR+/MCCFR/Deep CFR training")
    parser.add_argument("--stop-on-failure", action="store_true",
                        help="Stop immediately if any algorithm fails")

    # MCCFR-specific
    parser.add_argument("--mccfr-no-plus", action="store_true",
                        help="MCCFR: disable CFR+ regret floor (use vanilla regret).")

    # Deep CFR-specific
    parser.add_argument("--deep-cfr-iterations", type=int, default=None,
                        help="Outer iterations for Deep CFR (default: 200).")
    parser.add_argument("--traversals", type=int, default=None,
                        help="Tree traversals per Deep CFR iteration (default: 100).")
    parser.add_argument("--deep-cfr-lr", type=float, default=None,
                        help="Learning rate for Deep CFR networks (default: 0.001).")
    parser.add_argument("--deep-cfr-batch-size", type=int, default=None,
                        help="Batch size for Deep CFR network training (default: 2048).")
    parser.add_argument("--deep-cfr-train-steps", type=int, default=None,
                        help="SGD steps per Deep CFR network update (default: 4000).")

    # Self-play / opponent pool options (DQN/NFSP only)
    parser.add_argument("--self-play", action="store_true",
                        help="Enable self-play with opponent pool and curriculum (DQN/NFSP).")
    parser.add_argument("--loss-multiplier", type=float, default=None,
                        help="Reward shaping: scale losses by this factor (default: 1.5).")
    parser.add_argument("--no-reward-shaping", action="store_true",
                        help="Disable reward shaping (use raw payoffs only).")
    parser.add_argument("--snapshot-interval", type=int, default=None,
                        help="Episodes between opponent pool snapshots (0 = auto).")

    # Noise / realistic play options (DQN/NFSP only)
    parser.add_argument("--noise", action="store_true",
                        help="Enable action-biased exploration for realistic imperfect play.")
    parser.add_argument("--noise-rate", type=float, default=None,
                        help="Initial noise rate (default: 0.08, decays over training).")
    parser.add_argument("--aggression-bias", type=float, default=None,
                        help="Probability of aggressive action when exploring (default: 0.6).")
    parser.add_argument("--min-noise", type=float, default=None,
                        help="Minimum noise rate at end of training (default: 0.02).")

    # Stack-aware reward options (DQN/NFSP only)
    parser.add_argument("--stack-rewards", action="store_true",
                        help="Enable stack-aware BB-relative reward shaping (equity/bankroll).")
    parser.add_argument("--initial-stack-bb", type=float, default=None,
                        help="Initial stack in big blinds for stack tracking (default: 100).")
    parser.add_argument("--rebuy-penalty", type=float, default=None,
                        help="Reward penalty per rebuy, escalating (default: 0.15). Higher = harsher.")

    args, extra = parser.parse_known_args()

    # Expand "all" into the full list
    algos = []
    for a in args.algorithms:
        if a == "all":
            algos = ALGORITHMS[:]
            break
        if a not in algos:
            algos.append(a)

    log.info(f"Training pipeline: {', '.join(a.upper() for a in algos)}")

    results = {}
    wall_start = time.perf_counter()

    for algo in algos:
        algo_args = build_args(algo, args, extra)
        success = run_training(algo, algo_args)
        results[algo] = success

        if not success and args.stop_on_failure:
            log.error(f"Stopping pipeline due to {algo.upper()} failure (--stop-on-failure)")
            break

    total_elapsed = time.perf_counter() - wall_start

    # Summary
    log.info("")
    log.info(f"{'=' * 60}")
    log.info(f"Training Summary (total: {total_elapsed / 60:.1f} min)")
    log.info(f"{'=' * 60}")
    for algo, success in results.items():
        status = "OK" if success else "FAILED"
        model_dir = args.save_dir or f"./models/{algo}"
        log.info(f"  {algo.upper():6s}  [{status}]  -> {model_dir}")

    log.info("")
    log.info("To serve a trained model:")
    log.info("  set RLCARD_AGENT_TYPE=<dqn|nfsp|cfr|cfr_plus|mccfr|deep_cfr>")
    log.info("  set RLCARD_MODEL_PATH=./models/<algo>/")
    log.info("  python rlcard_server.py")

    # Exit with error if any failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
