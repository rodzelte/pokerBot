"""
train_cfr.py — Train rlcard's CFR (Counterfactual Regret Minimisation) agent for Hold'em.

CFR is a tabular algorithm that converges to a Nash equilibrium in two-player
zero-sum games by iterating over every information set.  Because the state
space of full NL Hold'em is enormous, this script defaults to Limit Hold'em
where CFR is tractable.  You can switch to NL Hold'em with --game, but expect
much longer training and higher memory usage.

Usage:
  python train_cfr.py                                       # 5k iterations on limit-holdem
  python train_cfr.py --iterations 20000                    # more iterations
  python train_cfr.py --game no-limit-holdem --iterations 2000  # NL (slow, large state space)
  python train_cfr.py --resume ./models/cfr                 # resume from checkpoint

The trained model is saved to ./models/cfr/ and can be loaded by the server:
  set RLCARD_MODEL_PATH=./models/cfr/
  set RLCARD_AGENT_TYPE=cfr
  python rlcard_server.py
"""
import argparse
import os
import logging
import signal
import time
import pickle

import rlcard
from rlcard.agents import CFRAgent, RandomAgent
from rlcard.utils import (
    set_seed,
    tournament,
)

# ── NumPy compat: rlcard 1.1.0 CFRAgent.get_state() calls ndarray.tostring()
# which was removed in NumPy 2.0.  Patch the method to use tobytes() instead.
import numpy as np
if not hasattr(np.ndarray(0), 'tostring'):
    _orig_get_state = CFRAgent.get_state
    def _patched_get_state(self, player_id):
        state = self.env.get_state(player_id)
        return state['obs'].tobytes(), list(state['legal_actions'].keys())
    CFRAgent.get_state = _patched_get_state

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("train_cfr")


def save_checkpoint(agent: CFRAgent, save_dir: str, iteration: int) -> None:
    """Persist CFR policy and regret tables."""
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, "cfr_model.pkl")
    data = {
        "regrets": agent.regrets,
        "cumulative_strategy": agent.cumulative_strategy if hasattr(agent, "cumulative_strategy") else {},
        "average_policy": agent.average_policy if hasattr(agent, "average_policy") else {},
        "iteration": agent.iteration,
        "episode": iteration,
        "agent_type": "cfr",
    }
    # Also try to save the policy table directly
    if hasattr(agent, "policy"):
        data["policy"] = agent.policy

    with open(checkpoint_path, "wb") as f:
        pickle.dump(data, f)


def load_checkpoint(agent: CFRAgent, save_dir: str) -> int:
    """Load CFR policy and regret tables. Returns the last iteration number."""
    checkpoint_path = os.path.join(save_dir, "cfr_model.pkl")
    if not os.path.exists(checkpoint_path):
        return 0
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)

    if "regrets" in data:
        agent.regrets = data["regrets"]
    if "cumulative_strategy" in data and hasattr(agent, "cumulative_strategy"):
        agent.cumulative_strategy = data["cumulative_strategy"]
    if "average_policy" in data and hasattr(agent, "average_policy"):
        agent.average_policy = data["average_policy"]
    if "policy" in data and hasattr(agent, "policy"):
        agent.policy = data["policy"]
    if "iteration" in data:
        agent.iteration = data["iteration"]

    return data.get("episode", data.get("iteration", 0))


def main():
    parser = argparse.ArgumentParser(description="Train rlcard CFR agent for Hold'em")
    parser.add_argument("--game", type=str, default="limit-holdem",
                        choices=["limit-holdem", "no-limit-holdem", "leduc-holdem"],
                        help="Game variant. CFR is tractable on limit/leduc, very slow on NL.")
    parser.add_argument("--iterations", type=int, default=5000,
                        help="Number of CFR iterations (each traverses the full game tree)")
    parser.add_argument("--eval-every", type=int, default=1000, help="Evaluate every N iterations")
    parser.add_argument("--eval-num", type=int, default=1000, help="Number of evaluation games")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="./models/cfr")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint directory")
    parser.add_argument("--log-every", type=int, default=500,
                        help="Log progress every N iterations.")
    args = parser.parse_args()

    set_seed(args.seed)

    if args.game == "no-limit-holdem":
        log.warning(
            "CFR on full NL Hold'em has an enormous state space. "
            "This will be very slow and memory-intensive. "
            "Consider using limit-holdem or leduc-holdem, or use NFSP/DQN for NL."
        )

    # Create environment
    env = rlcard.make(args.game, config={
        "game_num_players": 2,
        "seed": args.seed,
        "allow_step_back": True,
    })
    eval_env = rlcard.make(args.game, config={
        "game_num_players": 2,
        "seed": args.seed + 1,
        "allow_step_back": True,
    })

    # Create CFR agent — rlcard's CFR expects the env directly
    agent = CFRAgent(env, model_path=args.save_dir)

    # Resume from checkpoint
    resume_path = args.resume or args.save_dir
    start_iteration = 1

    saved_iter = load_checkpoint(agent, resume_path)
    if saved_iter > 0:
        start_iteration = saved_iter + 1
        log.info(f"Resumed CFR from iteration {saved_iter}")
        if start_iteration > args.iterations:
            log.info("Training already complete!")
            return
    else:
        log.info(f"Starting fresh CFR training on {args.game}.")

    # Random opponent for evaluation
    random_agent = RandomAgent(num_actions=env.num_actions)

    eval_env.set_agents([agent, random_agent])

    os.makedirs(args.save_dir, exist_ok=True)

    log.info(f"Training CFR iterations {start_iteration} to {args.iterations} on {args.game}...")

    # Graceful shutdown
    _shutdown_requested = False
    _current_iter = start_iteration

    def _handle_interrupt(signum, frame):
        nonlocal _shutdown_requested
        if _shutdown_requested:
            log.warning("Second interrupt — force exiting without save.")
            raise SystemExit(1)
        _shutdown_requested = True
        log.info(f"\nInterrupt received at iteration {_current_iter}. Saving checkpoint...")
        save_checkpoint(agent, args.save_dir, _current_iter)
        log.info(f"Checkpoint saved (iteration {_current_iter})")
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _handle_interrupt)

    # Performance log
    perf_path = os.path.join(args.save_dir, "performance.csv")
    write_header = not os.path.exists(perf_path) or os.path.getsize(perf_path) == 0

    with open(perf_path, "a") as perf_file:
        if write_header:
            perf_file.write("iteration,reward\n")

        wall_start = time.perf_counter()
        window_start = wall_start

        for iteration in range(start_iteration, args.iterations + 1):
            _current_iter = iteration

            # One CFR traversal of the game tree
            agent.train()

            # Manual save trigger
            trigger_path = os.path.join(args.save_dir, "SAVE")
            if os.path.exists(trigger_path):
                os.remove(trigger_path)
                save_checkpoint(agent, args.save_dir, iteration)
                log.info(f"Manual save triggered at iteration {iteration}.")

            if args.log_every > 0 and iteration % args.log_every == 0:
                elapsed = max(time.perf_counter() - window_start, 1e-9)
                iters_per_sec = args.log_every / elapsed
                total_elapsed = max(time.perf_counter() - wall_start, 1e-9)
                info_sets = len(agent.regrets) if hasattr(agent, "regrets") else 0
                pct = iteration / args.iterations * 100
                remaining = (args.iterations - iteration) / max(iters_per_sec, 0.01)
                log.info(
                    f"[CFR] {iteration}/{args.iterations} ({pct:.1f}%) | "
                    f"Speed: {iters_per_sec:.1f} iter/s | "
                    f"Info sets: {info_sets} | "
                    f"Elapsed: {total_elapsed / 60:.1f} min | "
                    f"ETA: {remaining / 60:.1f} min"
                )
                window_start = time.perf_counter()

            # Periodic evaluation + checkpoint
            if iteration % args.eval_every == 0:
                reward = tournament(eval_env, args.eval_num)[0]
                perf_file.write(f"{iteration},{reward}\n")
                perf_file.flush()
                pct = iteration / args.iterations * 100
                log.info(
                    f"[CFR] Checkpoint {iteration}/{args.iterations} ({pct:.1f}%) | "
                    f"Eval reward: {reward:.4f}"
                )
                save_checkpoint(agent, args.save_dir, iteration)
                log.info(f"[CFR] Saved -> {args.save_dir}")

        if args.iterations % args.eval_every != 0:
            save_checkpoint(agent, args.save_dir, args.iterations)
            log.info(f"Final checkpoint saved (iteration {args.iterations})")

    log.info("CFR Training complete!")
    log.info(f"Model saved to {args.save_dir}")
    log.info(f"To use: set RLCARD_MODEL_PATH={args.save_dir} & set RLCARD_AGENT_TYPE=cfr")


if __name__ == "__main__":
    main()
