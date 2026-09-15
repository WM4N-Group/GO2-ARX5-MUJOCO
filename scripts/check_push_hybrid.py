"""Check hybrid action routing, physical contact preparation and resets."""

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--output-state-json", type=Path)
parser.add_argument("--box-height", type=float, default=0.25)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output_state_json is not None and args.output_state_json.exists():
    parser.error("State output must not already exist")
if not 0.0 < args.box_height < 1.0:
    parser.error("Box height must be between zero and one metre")
launcher = AppLauncher(args)

import gymnasium as gym
import torch

import isaaclab_tasks
from isaaclab_tasks.utils import load_cfg_from_registry

import LeggedManip_Lab.tasks


def main():
    config = load_cfg_from_registry("GO2-ARX5-Box-Push-Hybrid-Play", "env_cfg_entry_point")
    config.scene.num_envs = 4
    config.sim.device = args.device
    config.seed = 100
    config.box_size = (*config.box_size[:2], args.box_height)
    config.scene.box.spawn.size = config.box_size
    config.scene.box.init_state.pos = (*config.scene.box.init_state.pos[:2], args.box_height / 2.0)
    env = gym.make("GO2-ARX5-Box-Push-Hybrid-Play", cfg=config).unwrapped
    observation, _info = env.reset()
    assert observation["policy"].shape == (4, 210)
    assert observation["box"].shape == (4, 16)
    assert env.action_manager.total_action_dim == 12
    term = env.action_manager.get_term("joint_pos")
    assert term.controller.leg_slots == list(range(12))
    assert term.controller.action_slots == list(range(12, 18))
    requested = torch.linspace(-0.03, 0.03, 12, device=args.device).expand(4, -1).clone()
    term.process_actions(requested)
    torch.testing.assert_close(term.raw_actions, requested)
    torch.testing.assert_close(term.joint_commands[:, :12], torch.zeros_like(requested))
    assert torch.isfinite(term.processed_actions).all()
    env.reset()
    ready = torch.zeros(4, dtype=torch.bool, device=args.device)
    failed = torch.zeros_like(ready)
    snapshots = []
    robot = env.scene["robot"]
    controller = term.controller
    for step in range(300):
        if args.output_state_json is not None and step % 25 == 0:
            gravity = robot.root_physx_view.get_gravity_compensation_forces()
            columns = controller.joint_ids if gravity.shape[1] == robot.num_joints else controller.jacobian_columns
            jacobian = robot.root_physx_view.get_jacobians()[:, controller.jacobian_body][:, :, controller.jacobian_columns]
            snapshots.append({
                "step": step, "root_state": robot.data.root_state_w.cpu().tolist(),
                "body_position": robot.data.body_pos_w.cpu().tolist(),
                "body_quaternion": robot.data.body_quat_w.cpu().tolist(),
                "joint_pos": robot.data.joint_pos.cpu().tolist(),
                "joint_vel": robot.data.joint_vel.cpu().tolist(),
                "hand_position": robot.data.body_pos_w[:, controller.hand_id].cpu().tolist(),
                "mount_position": robot.data.body_pos_w[:, controller.mount_id].cpu().tolist(),
                "mount_quaternion": robot.data.body_quat_w[:, controller.mount_id].cpu().tolist(),
                "hand_jacobian": jacobian.cpu().tolist(), "arm_gravity": gravity[:, columns].cpu().tolist(),
                "arm_reference": controller.reference.cpu().tolist(), "locked": controller.locked.cpu().tolist(),
            })
        _obs, _reward, terminated, _timeout, _extra = env.step(torch.zeros((4, 12), device=args.device))
        ready |= term.controller.motion_ready
        failed |= terminated & ~env.termination_manager.get_term("box_settled")
    if args.output_state_json is not None:
        report = {
            "joint_names": robot.joint_names, "body_names": robot.body_names,
            "body_masses": robot.root_physx_view.get_masses().cpu().tolist(),
            "body_coms_xyzw": robot.root_physx_view.get_coms().cpu().tolist(),
            "body_inertias": robot.root_physx_view.get_inertias().cpu().tolist(),
            "default_joint_pos": robot.data.default_joint_pos.cpu().tolist(),
            "soft_joint_limits": robot.data.soft_joint_pos_limits.cpu().tolist(),
            "joint_stiffness": robot.data.joint_stiffness.cpu().tolist(),
            "joint_damping": robot.data.joint_damping.cpu().tolist(),
            "box_height": args.box_height, "ready": ready.cpu().tolist(), "failed": failed.cpu().tolist(),
            "snapshots": snapshots,
        }
        args.output_state_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_state_json.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
    assert ready.all() and not failed.any()
    term.process_actions(requested)
    active = term.controller.motion_ready
    torch.testing.assert_close(term.joint_commands[active, :12], requested[active])
    preserved = term.controller.motion_ready[[1, 3]].clone()
    term.reset(torch.tensor([0, 2], device=args.device))
    assert not term.controller.motion_ready[[0, 2]].any()
    assert not env.box_push_pose_override_active[[0, 2]].any()
    torch.testing.assert_close(term.controller.motion_ready[[1, 3]], preserved)
    torch.testing.assert_close(term.joint_commands[[0, 2]], torch.zeros((2, 18), device=args.device))
    print("HYBRID_CONTRACT_PASSED: 226 observations, 12 learned actions, 18 applied commands, 4/4 contact preparation, isolated resets", flush=True)
    env.close()


try:
    main()
finally:
    launcher.app.close()