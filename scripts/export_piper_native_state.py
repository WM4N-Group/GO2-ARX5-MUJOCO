"""Export matched PIPER physics states for MuJoCo adapter validation."""

import argparse
import hashlib
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-json", type=Path, required=True)
parser.add_argument("--samples", type=int, default=16)
parser.add_argument("--task", choices=("GO2-PIPER-Box-Push-Hybrid-Play", "GO2-PIPER-Box-Climb-Play"), default="GO2-PIPER-Box-Push-Hybrid-Play")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output_json.exists() or args.samples < 1:
    parser.error("Require a new output and positive sample count")
launcher = AppLauncher(args)

import gymnasium as gym
import torch
import isaaclab_tasks
from isaaclab_tasks.utils import load_cfg_from_registry
import LeggedManip_Lab.tasks


def main():
    config = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    config.scene.num_envs = 1
    config.sim.device = args.device
    config.seed = 60
    config.observations.policy.enable_corruption = False
    if "Climb" in args.task:
        config.events.add_base_mass = None
    env = gym.make(args.task, cfg=config).unwrapped
    try:
        env.reset()
        robot = env.scene["robot"]
        generator = torch.Generator(device=args.device).manual_seed(60)
        hand = robot.find_bodies("end_effector")[0][0]
        samples = []
        for _sample in range(args.samples):
            state = robot.data.default_root_state.clone()
            state[:, :3] = env.scene.env_origins + torch.tensor([0, 0, 1.2], device=args.device)
            state[:, 3:7] = torch.tensor([1, 0, 0, 0], device=args.device)
            state[:, 7:] = 0
            limits = robot.data.soft_joint_pos_limits
            position = limits[..., 0] + (limits[..., 1] - limits[..., 0]) * torch.rand(robot.data.joint_pos.shape, generator=generator, device=args.device)
            joint_velocity = torch.zeros_like(position)
            if "Climb" in args.task:
                state[:, 2] = env.scene.env_origins[:, 2] + 0.33
                state[:, 7:] = torch.rand(state[:, 7:].shape, generator=generator, device=args.device) * 0.6 - 0.3
                position = torch.clamp(robot.data.default_joint_pos + torch.rand(position.shape, generator=generator, device=args.device) * 0.04 - 0.02, min=limits[..., 0], max=limits[..., 1])
                joint_velocity = torch.rand(position.shape, generator=generator, device=args.device) * 2.0 - 1.0
            robot.write_root_state_to_sim(state)
            robot.write_joint_state_to_sim(position, joint_velocity)
            robot.set_joint_position_target(position)
            env.scene.write_data_to_sim()
            env.sim.forward()
            env.scene.update(0.0)
            if "Climb" in args.task:
                env.scene["height_scanner"].update(env.step_dt, force_recompute=True)
            observations = env.observation_manager.compute()
            samples.append({
                "root_position": (robot.data.root_pos_w[0] - env.scene.env_origins[0]).tolist(), "root_quaternion": robot.data.root_quat_w[0].tolist(),
                "root_linear_velocity_world": robot.data.root_lin_vel_w[0].tolist(),
                "root_angular_velocity_world": robot.data.root_ang_vel_w[0].tolist(),
                "joint_positions": robot.data.joint_pos[0].tolist(), "joint_velocities": robot.data.joint_vel[0].tolist(),
                "body_positions": (robot.data.body_pos_w[0] - env.scene.env_origins[0]).tolist(),
                "policy_observation": observations["policy"][0].tolist(),
                "body_quaternions": robot.data.body_quat_w[0].tolist(),
                "hand_jacobian": robot.root_physx_view.get_jacobians()[0, hand].tolist(),
                "gravity_forces": robot.root_physx_view.get_gravity_compensation_forces()[0].tolist(),
            })
        report = {
            "schema_version": 1, "task": args.task, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "joint_names": list(robot.joint_names), "body_names": list(robot.body_names),
            "masses": robot.root_physx_view.get_masses()[0].tolist(),
            "centers_of_mass": robot.root_physx_view.get_coms()[0].tolist(),
            "inertias": robot.root_physx_view.get_inertias()[0].tolist(), "samples": samples,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("x") as destination:
            json.dump(report, destination, indent=2, allow_nan=False)
        print(f"PIPER_STATES samples={len(samples)} output={args.output_json}", flush=True)
    finally:
        env.close()


try:
    main()
finally:
    launcher.app.close()