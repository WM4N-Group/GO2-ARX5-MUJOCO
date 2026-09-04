"""Evaluate an SB3 GO2-ARX5 checkpoint with the MuJoCo viewer."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

try:
    from stable_baselines3 import PPO
except ImportError as exc:
    raise SystemExit(
        "Training dependencies are missing. Run: "
        "python -m pip install -r mujoco/train/go2_arx5/requirements.txt"
    ) from exc

from go2_arx5_env import Go2ARX5Env


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a GO2-ARX5 MuJoCo PPO checkpoint."
    )
    parser.add_argument("checkpoint", type=Path, help="SB3 .zip checkpoint")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--command-profile",
        choices=("train", "full", "stand"),
        default="train",
    )
    args = parser.parse_args()

    env = Go2ARX5Env(
        render_mode="human", command_profile=args.command_profile
    )
    model = PPO.load(args.checkpoint, device="cpu")
    try:
        for episode in range(args.episodes):
            observation, _ = env.reset(seed=episode)
            terminated = truncated = False
            episode_reward = 0.0
            while not (terminated or truncated):
                start = time.perf_counter()
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(
                    action
                )
                episode_reward += reward
                sleep_time = env.control_dt - (time.perf_counter() - start)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            print(f"Episode {episode + 1}: reward={episode_reward:.3f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
