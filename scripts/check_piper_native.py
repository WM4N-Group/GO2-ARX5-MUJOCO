"""Evaluate an existing PIPER actor in its native Isaac task without training."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", choices=("GO2-PIPER-Flat", "GO2-PIPER-WBC"), default="GO2-PIPER-Flat")
parser.add_argument("--policy", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--duration", type=float, default=10.0)
parser.add_argument("--settle", type=float, default=3.0)
parser.add_argument("--velocity", type=float, nargs=3, default=(0.25, 0.0, 0.0))
parser.add_argument("--ee-position", type=float, nargs=3, default=(0.5, 0.0, 0.4))
parser.add_argument("--output-json", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output_json.exists():
    parser.error("Refusing to overwrite an existing report")
if args.num_envs < 1 or args.seed < 0 or args.duration <= 0 or args.settle < 0:
    parser.error("Invalid environment count, seed or duration")
if not all(math.isfinite(value) for value in (args.duration, args.settle, *args.velocity, *args.ee_position)):
    parser.error("All numeric parameters must be finite")
launcher = AppLauncher(args)

import gymnasium as gym
import torch

import isaaclab_tasks
from isaaclab_tasks.utils import load_cfg_from_registry

import LeggedManip_Lab
import LeggedManip_Lab.tasks


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    source_root = Path(__file__).resolve().parents[1]
    package_path = Path(LeggedManip_Lab.__file__).resolve()
    if not package_path.is_relative_to(source_root):
        raise RuntimeError(f"Imported another checkout: {package_path}")
    config = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    config.scene.num_envs = args.num_envs
    config.sim.device = args.device
    config.seed = args.seed
    config.episode_length_s = args.settle + args.duration + 5.0
    config.observations.policy.enable_corruption = False
    velocity = config.commands.base_velocity
    velocity.rel_standing_envs = 0.0
    velocity.heading_command = False
    velocity.resampling_time_range = (10000.0, 10000.0)
    velocity.curriculum_enabled = False
    for ranges in (velocity.ranges, velocity.limit_ranges):
        for name, value in zip(("lin_vel_x", "lin_vel_y", "ang_vel_z"), args.velocity):
            setattr(ranges, name, (value, value))
    pose = config.commands.ee_pose
    pose.resampling_time_range = (10000.0, 10000.0)
    pose.curriculum_enabled = False
    for ranges in (pose.ranges, pose.limit_ranges):
        for name, value in zip(("pos_x", "pos_y", "pos_z", "roll", "pitch", "yaw"), (*args.ee_position, 0.0, 0.0, 0.0)):
            setattr(ranges, name, (value, value))
    env = gym.make(args.task, cfg=config).unwrapped
    try:
        observations, _extras = env.reset()
        robot = env.scene["robot"]
        policy = torch.jit.load(str(args.policy), map_location=args.device).eval()
        active = torch.ones(args.num_envs, dtype=torch.bool, device=args.device)
        terminated = torch.zeros_like(active)
        timed_out = torch.zeros_like(active)
        posture_failure = torch.zeros_like(active)
        first_failure_step = torch.full((args.num_envs,), -1, dtype=torch.long, device=args.device)
        minimum_height = torch.full((args.num_envs,), torch.inf, device=args.device)
        velocity_sum = torch.zeros((args.num_envs, 3), device=args.device)
        samples = torch.zeros(args.num_envs, device=args.device)
        settle_steps = round(args.settle / env.step_dt)
        policy_steps = round(args.duration / env.step_dt)
        commanded_velocity = torch.tensor(args.velocity, device=args.device)
        commanded_pose = env.command_manager.get_command("ee_pose").clone()
        masses = robot.root_physx_view.get_masses().cpu().tolist()
        initial_defaults = robot.data.default_joint_pos[0].cpu().tolist()
        for step in range(settle_steps + policy_steps):
            if observations["policy"].shape != (args.num_envs, 210):
                raise RuntimeError(f"Unexpected policy input: {observations['policy'].shape}")
            with torch.inference_mode():
                actions = torch.zeros((args.num_envs, 18), device=args.device) if step < settle_steps else policy(observations["policy"]).clamp(-20.0, 20.0)
            if actions.shape != (args.num_envs, 18) or not torch.isfinite(actions).all():
                raise RuntimeError("Invalid native policy actions")
            observations, _reward, failures, timeouts, _extras = env.step(actions)
            ended = active & (failures | timeouts)
            first_failure_step[ended] = step
            terminated |= active & failures
            timed_out |= active & timeouts
            active &= ~(failures | timeouts)
            heights = robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
            minimum_height[active] = torch.minimum(minimum_height[active], heights[active])
            unstable = active & ((heights < 0.16) | (robot.data.projected_gravity_b[:, 2] > -0.5))
            first_failure_step[unstable] = step
            posture_failure |= unstable
            active &= ~unstable
            if step >= settle_steps:
                measured = torch.cat((robot.data.root_lin_vel_b[:, :2], robot.data.root_ang_vel_b[:, 2:3]), dim=-1)
                velocity_sum[active] += measured[active]
                samples[active] += 1
        mean_velocity = velocity_sum / samples.clamp_min(1).unsqueeze(-1)
        tracking = (torch.linalg.vector_norm(mean_velocity[:, :2] - commanded_velocity[:2], dim=-1) <= max(0.12, 0.5 * math.hypot(*args.velocity[:2])))
        tracking &= (mean_velocity[:, 2] - commanded_velocity[2]).abs() <= 0.2
        passed = active & tracking & (samples == policy_steps)
        package_root = package_path.parent
        assets = sorted((package_root / "assets/go2_piper").rglob("*.usd"))
        sources = sorted((package_root / "tasks/manager_based/leggedmanip_lab").rglob("*.py"))
        sources.extend((package_root / "assets/go2_piper/go2_piper_articulation_cfg.py", Path(__file__)))
        report = {
            "schema_version": 1,
            "scope": "native PIPER actor diagnostic; first episodes only; not hardware acceptance",
            "task": args.task, "seed": args.seed, "num_envs": args.num_envs,
            "package_path": str(package_path), "policy_sha256": sha256(args.policy),
            "source_sha256": {str(path): sha256(path) for path in sources},
            "usd_sha256": {str(path): sha256(path) for path in assets},
            "velocity_command": args.velocity, "ee_position_argument": args.ee_position,
            "native_pose_command": commanded_pose.cpu().tolist(),
            "control_dt": env.step_dt, "settle_steps": settle_steps, "policy_steps": policy_steps,
            "active_termination_terms": list(env.termination_manager.active_terms),
            "successful_episodes": int(passed.sum().item()), "passed": passed.cpu().tolist(),
            "terminated": terminated.cpu().tolist(), "timed_out": timed_out.cpu().tolist(),
            "posture_failure": posture_failure.cpu().tolist(), "first_failure_step": first_failure_step.cpu().tolist(),
            "minimum_height": [value if math.isfinite(value) else None for value in minimum_height.cpu().tolist()],
            "mean_body_velocity": mean_velocity.cpu().tolist(), "policy_samples": samples.cpu().tolist(),
            "joint_names": list(robot.joint_names), "body_names": list(robot.body_names),
            "default_joint_positions": initial_defaults, "body_masses_per_environment": masses,
            "torch": torch.__version__, "device": str(args.device),
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("x") as destination:
            json.dump(report, destination, indent=2, allow_nan=False)
        print("PIPER_NATIVE=" + json.dumps({"successful_episodes": report["successful_episodes"], "num_envs": args.num_envs, "output": str(args.output_json)}), flush=True)
        return 0 if passed.all() else 1
    finally:
        env.close()


try:
    result = main()
finally:
    launcher.app.close()
raise SystemExit(result)