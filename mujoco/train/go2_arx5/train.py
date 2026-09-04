"""Train GO2-ARX5 with Stable-Baselines3 PPO and MuJoCo."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        CheckpointCallback,
        EvalCallback,
    )
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv
except ImportError as exc:
    raise SystemExit(
        "Training dependencies are missing. Run: "
        "python -m pip install -r mujoco/train/go2_arx5/requirements.txt"
    ) from exc

from go2_arx5_env import Go2ARX5Env


class DeterministicActor(nn.Module):
    """Deployment wrapper matching the existing 210 -> 18 interface."""

    def __init__(self, policy: nn.Module) -> None:
        super().__init__()
        self.features_extractor = policy.features_extractor
        self.mlp_extractor = policy.mlp_extractor
        self.action_net = policy.action_net

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        features = self.features_extractor(observation)
        latent = self.mlp_extractor.forward_actor(features)
        return torch.clamp(self.action_net(latent), -10.0, 10.0)


class RewardTermsCallback(BaseCallback):
    """Write mean reward components from all environments to TensorBoard."""

    def _on_step(self) -> bool:
        values: dict[str, list[float]] = {}
        for info in self.locals.get("infos", []):
            for name, value in info.get("reward_terms", {}).items():
                values.setdefault(name, []).append(value)
        for name, samples in values.items():
            self.logger.record_mean(f"reward_terms/{name}", sum(samples) / len(samples))
        return True


def export_policy(model: PPO, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    actor = DeterministicActor(model.policy).to("cpu").eval()
    scripted = torch.jit.trace(
        actor, torch.zeros(1, 210, dtype=torch.float32)
    )
    scripted.save(str(output_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train GO2-ARX5 directly in MuJoCo."
    )
    parser.add_argument("--total-timesteps", type=int, default=5_000_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--run-dir", type=Path, default=Path("runs/mujoco/go2_arx5")
    )
    parser.add_argument(
        "--resume", type=Path, default=None, help="SB3 .zip checkpoint"
    )
    parser.add_argument(
        "--command-profile",
        choices=("train", "full", "stand"),
        default="train",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument(
        "--render",
        action="store_true",
        help="Show the training environment (requires --num-envs 1 and is slower)",
    )
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_envs < 1:
        raise SystemExit("--num-envs must be at least 1")
    if args.render and args.num_envs != 1:
        raise SystemExit("--render requires --num-envs 1")
    if args.batch_size > args.n_steps * args.num_envs:
        raise SystemExit("--batch-size cannot exceed --n-steps * --num-envs")
    args.run_dir.mkdir(parents=True, exist_ok=True)

    env_kwargs = {
        "command_profile": args.command_profile,
        "render_mode": "human" if args.render else None,
    }
    vec_env_cls = SubprocVecEnv if args.num_envs > 1 else None
    train_env = make_vec_env(
        Go2ARX5Env,
        n_envs=args.num_envs,
        seed=args.seed,
        env_kwargs=env_kwargs,
        vec_env_cls=vec_env_cls,
        vec_env_kwargs={"start_method": "forkserver"} if vec_env_cls else None,
    )
    eval_env = make_vec_env(
        Go2ARX5Env,
        n_envs=1,
        seed=args.seed + 10_000,
        env_kwargs={"command_profile": args.command_profile},
    )

    if args.resume:
        model = PPO.load(args.resume, env=train_env, device=args.device)
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            policy_kwargs={
                "activation_fn": nn.ELU,
                "net_arch": {
                    "pi": [512, 256, 128],
                    "vf": [512, 256, 128],
                },
            },
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,
            max_grad_norm=1.0,
            target_kl=args.target_kl,
            tensorboard_log=str(args.run_dir / "tensorboard"),
            seed=args.seed,
            device=args.device,
            verbose=1,
        )

    callbacks = [
        RewardTermsCallback(),
        CheckpointCallback(
            save_freq=max(100_000 // args.num_envs, 1),
            save_path=str(args.run_dir / "checkpoints"),
            name_prefix="go2_arx5",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(args.run_dir / "best"),
            log_path=str(args.run_dir / "eval"),
            eval_freq=max(100_000 // args.num_envs, 1),
            n_eval_episodes=5,
            deterministic=True,
        ),
    ]

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            reset_num_timesteps=args.resume is None,
        )
        model.save(args.run_dir / "final_model")
        export_policy(model, args.run_dir / "exported/policy.pt")
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
