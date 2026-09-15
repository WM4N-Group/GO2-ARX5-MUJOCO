"""Check whether the box-task robot can hold its nominal standing pose."""

import argparse
import hashlib
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=("GO2-ARX5-Box-Climb-Play", "GO2-ARX5-Box-Push-Play", "GO2-PIPER-Box-Climb-Play", "GO2-PIPER-Box-Push-Hybrid-Play"), default="GO2-ARX5-Box-Climb-Play")
parser.add_argument("--output-json", type=Path, required=True)
parser.add_argument("--spawn-height", type=float, default=0.33)
parser.add_argument("--prepared-starts", type=Path)
parser.add_argument("--prepared-phase", choices=("ground", "platform"), default="ground")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output_json.exists():
    parser.error("Output already exists")
launcher = AppLauncher(args)

import gymnasium as gym
import torch

import isaaclab_tasks
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.utils import load_cfg_from_registry

import LeggedManip_Lab.tasks


def main():
    config = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    config.scene.num_envs = 16
    config.sim.device = args.device
    config.seed = 100
    config.scene.robot.init_state.pos = (0.0, 0.0, args.spawn_height)
    if args.prepared_starts is not None:
        if "Climb" not in args.task:
            raise ValueError("Prepared starts require the CLIMB task")
        config.prepared_start_path = str(args.prepared_starts)
        config.prepared_start_phase = args.prepared_phase
        if args.prepared_phase == "platform":
            terrain = config.scene.terrain.terrain_generator.sub_terrains["box"]
            terrain.approach_height = 0.20
            terrain.approach_gap = 0.015
            terrain.box_length = 2.0
            terrain.box_width = 1.6
            terrain.box_height_range = (0.20, 0.20)
    env = gym.make(args.task, cfg=config).unwrapped
    initial_observations, _extras = env.reset()
    robot = env.scene["robot"]
    actuator_limits = {
        name: {
            "effort_limit": actuator.effort_limit[0].cpu().tolist(),
            "velocity_limit": actuator.velocity_limit[0].cpu().tolist(),
            "stiffness": actuator.stiffness[0].cpu().tolist(),
            "damping": actuator.damping[0].cpu().tolist(),
        }
        for name, actuator in robot.actuators.items()
    }
    if args.prepared_starts is not None:
        from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.mdp.climb_start_states import load_start_states

        names, states = load_start_states(args.prepared_starts, args.prepared_phase, args.device)
        joint_ids, _names = robot.find_joints(names, preserve_order=True)
        errors = (robot.data.joint_pos[:, None, joint_ids] - states["joint_position"][None]).abs().amax(dim=-1).amin(dim=1)
        if (errors > 1e-5).any():
            raise AssertionError("Prepared joint positions were not preserved at reset")
        arm_ids, _names = robot.find_joints("joint.*")
        torch.testing.assert_close(robot.data.default_joint_pos[:, arm_ids], torch.zeros((16, 6), device=args.device))
    feet = SceneEntityCfg("robot", body_names=".*_foot")
    feet.resolve(env.scene)
    sensor = SceneEntityCfg("contact_forces", body_names=".*_foot")
    sensor.resolve(env.scene)
    initial_feet = robot.data.body_pos_w[:, feet.body_ids, 2] - env.scene.env_origins[:, 2, None]
    initial_feet = initial_feet.clone()
    terminated = torch.zeros(16, dtype=torch.bool, device=args.device)
    minimum_height = torch.full((16,), torch.inf, device=args.device)
    upright = torch.ones(16, dtype=torch.bool, device=args.device)
    hand_contact_seen = torch.zeros_like(terminated)
    if "Push" in args.task:
        from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.mdp import box_push

        hand = SceneEntityCfg("robot", body_names="end_effector")
        hand.resolve(env.scene)
    for step in range(200):
        _observation, _reward, failures, timeouts, _extras = env.step(torch.zeros((16, env.action_manager.total_action_dim), device=args.device))
        terminated |= failures | timeouts
        if "Push" in args.task:
            hand_contact_seen |= box_push.hand_contact(env, hand)
        if step >= 50:
            relative_height = robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
            minimum_height = torch.minimum(minimum_height, relative_height)
            upright &= robot.data.projected_gravity_b[:, 2] < -0.94
    support = env.scene["contact_forces"].data.net_forces_w[:, sensor.body_ids, 2] > 2.0
    passed = ~terminated & upright & (minimum_height > 0.20) & (support.sum(dim=1) >= 3)
    result = {
        "task": args.task,
        "robot_usd": config.scene.robot.spawn.usd_path,
        "observation_shapes": {name: list(value.shape) for name, value in initial_observations.items()},
        "action_size": env.action_manager.total_action_dim,
        "actuator_limits": actuator_limits,
        "joint_names": list(robot.joint_names),
        "simulation_effort_limits": robot.data.joint_effort_limits[0].cpu().tolist(),
        "body_names": list(robot.body_names),
        "body_masses": robot.root_physx_view.get_masses()[0].cpu().tolist(),
        "prepared_start_sha256": hashlib.sha256(args.prepared_starts.read_bytes()).hexdigest() if args.prepared_starts is not None else None,
        "prepared_phase": args.prepared_phase if args.prepared_starts is not None else None,
        "spawn_height": args.spawn_height, "initial_foot_height_range": [initial_feet.min().item(), initial_feet.max().item()],
        "passed": passed.tolist(), "success_count": int(passed.sum().item()), "env_count": 16,
        "minimum_height_after_one_second": minimum_height.tolist(), "terminated": terminated.tolist(),
        "supporting_feet_final": support.sum(dim=1).tolist(),
    }
    if "Push" in args.task:
        result["hand_contact_seen"] = hand_contact_seen.tolist()
        result["final_hand_target_error"] = torch.linalg.vector_norm(
            box_push.push_hand_target(env) - robot.data.body_pos_w[:, hand.body_ids[0]], dim=1,
        ).tolist()
        if "Hybrid" in args.task:
            result["motion_ready"] = env.action_manager.get_term("joint_pos").controller.motion_ready.tolist()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print("BOX_STARTUP_CHECK=" + json.dumps(result), flush=True)
    env.close()
    if not passed.all():
        raise AssertionError("Nominal pose is not stable in every environment")


try:
    main()
finally:
    launcher.app.close()