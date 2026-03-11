"""
train_dqn.py — Train rlcard's DQN agent for No-Limit Hold'em.

Usage:
  python train_dqn.py                          # 50k episodes, saves to ./models/
  python train_dqn.py --episodes 200000        # more episodes for better play
  python train_dqn.py --resume ./models/dqn    # resume from checkpoint

The trained model is saved to ./models/dqn/ and can be loaded by the server:
  set RLCARD_MODEL_PATH=./models/dqn/checkpoint.pt
  python rlcard_server.py
"""
import argparse
import os
import logging

import torch
import rlcard
from rlcard.agents import DQNAgent, RandomAgent
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("train")


def main():
    parser = argparse.ArgumentParser(description="Train rlcard DQN agent for NL Hold'em")
    parser.add_argument("--episodes", type=int, default=50000, help="Number of training episodes")
    parser.add_argument("--eval-every", type=int, default=5000, help="Evaluate every N episodes")
    parser.add_argument("--eval-num", type=int, default=1000, help="Number of evaluation games")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="./models/dqn")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    log.info(f"Device: {device}")

    # Create environment
    env = rlcard.make("no-limit-holdem", config={
        "game_num_players": 2,
        "seed": args.seed,
    })
    eval_env = rlcard.make("no-limit-holdem", config={
        "game_num_players": 2,
        "seed": args.seed + 1,
    })

    # Agent config for save/restore
    agent_config = dict(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        mlp_layers=[256, 256, 128],
        device=device,
        learning_rate=args.lr,
        replay_memory_size=50000,
        batch_size=64,
        update_target_estimator_every=2000,
        epsilon_decay_steps=30000,
        epsilon_start=1.0,
        epsilon_end=0.05,
    )

    # Create DQN agent — auto-resume from checkpoint if it exists
    checkpoint_path = os.path.join(args.save_dir, "checkpoint.pt")
    resume_path = args.resume or args.save_dir
    resume_ckpt = os.path.join(resume_path, "checkpoint.pt")
    start_episode = 1

    agent = DQNAgent(**agent_config)

    if os.path.exists(resume_ckpt):
        log.info(f"Found checkpoint: {resume_ckpt} — resuming...")
        ckpt = torch.load(resume_ckpt, map_location=device, weights_only=False)
        agent.q_estimator.qnet.load_state_dict(ckpt["q_net"])
        agent.target_estimator.qnet.load_state_dict(ckpt["target_net"])

        # Replay memory is NOT saved, so reset total_t to 0 to let the
        # buffer warm up again before training resumes.  The model weights
        # are the important part — training quality is unaffected.
        agent.total_t = 0

        # Recover episode count: prefer checkpoint value, fall back to
        # the last entry in performance.csv if the checkpoint is old-format.
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
        log.info("No checkpoint found — starting fresh.")

    # Random opponent for training
    random_agent = RandomAgent(num_actions=env.num_actions)

    env.set_agents([agent, random_agent])
    eval_env.set_agents([agent, random_agent])

    os.makedirs(args.save_dir, exist_ok=True)

    log.info(f"Training episodes {start_episode} to {args.episodes}...")

    # Append to performance log instead of overwriting
    perf_path = os.path.join(args.save_dir, "performance.csv")
    write_header = not os.path.exists(perf_path) or os.path.getsize(perf_path) == 0

    with open(perf_path, "a") as perf_file:
        if write_header:
            perf_file.write("episode,reward\n")

        for episode in range(start_episode, args.episodes + 1):
            # Generate training data by playing one game
            trajectories, payoffs = env.run(is_training=True)

            # Reorganize raw trajectories into (state, action, reward, next_state, done) tuples
            trajectories = reorganize(trajectories, payoffs)

            # Feed transitions to the agent
            for ts in trajectories[0]:
                agent.feed(ts)

            # Periodic evaluation
            if episode % args.eval_every == 0:
                reward = tournament(eval_env, args.eval_num)[0]
                perf_file.write(f"{episode},{reward}\n")
                perf_file.flush()
                log.info(f"Episode {episode}/{args.episodes} | Eval reward: {reward:.4f}")

                # Save checkpoint with episode count
                torch.save({
                    "q_net": agent.q_estimator.qnet.state_dict(),
                    "target_net": agent.target_estimator.qnet.state_dict(),
                    "total_t": agent.total_t,
                    "episode": episode,
                    "agent_config": agent_config,
                }, checkpoint_path)
                log.info(f"Checkpoint saved to {checkpoint_path} (episode {episode})")

    log.info("Training complete!")
    log.info(f"Model saved to {checkpoint_path}")
    log.info(f"To use: set RLCARD_MODEL_PATH={checkpoint_path}")


if __name__ == "__main__":
    main()
