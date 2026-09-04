"""Run deterministic checks on the GO2-ARX5 Gymnasium environment."""

from __future__ import annotations

import numpy as np
from gymnasium.utils.env_checker import check_env

from go2_arx5_env import Go2ARX5Env


def main() -> None:
    env = Go2ARX5Env(command_profile="stand")
    try:
        check_env(env, skip_render_check=True)
        observation, _ = env.reset(seed=0)
        total_reward = 0.0
        reset_count = 0
        for _ in range(200):
            observation, reward, terminated, truncated, _ = env.step(
                np.zeros(18, dtype=np.float32)
            )
            total_reward += reward
            if terminated or truncated:
                reset_count += 1
                observation, _ = env.reset()
        assert observation.shape == (210,)
        assert np.isfinite(observation).all()
        print(
            "Environment check passed; "
            f"200-step reward={total_reward:.3f}, resets={reset_count}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
