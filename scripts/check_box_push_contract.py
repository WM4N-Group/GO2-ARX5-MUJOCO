"""Check the pretrained arm frame against the physical pushing target."""

import argparse
from types import SimpleNamespace

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)

import gymnasium as gym
import torch

import isaaclab_tasks
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz
from isaaclab_tasks.utils import load_cfg_from_registry

import LeggedManip_Lab.tasks
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.mdp import box_push


def main():
    yaw = torch.tensor([1.2], device=args.device)
    quaternion = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    mount_position = torch.tensor([[[0.2, -0.1, 0.42]]], device=args.device)
    box_position = torch.tensor([[1.2, 0.0, 0.125]], device=args.device)
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=args.device)
    fake = SimpleNamespace(
        num_envs=1, device=args.device, cfg=SimpleNamespace(box_size=(1.2, 1.2, 0.25), push_press_depth=0.02, push_approach_distance=0.10),
        box_push_hand_id=1, box_push_target=torch.tensor([[1.8, 0.0, 0.125]], device=args.device),
        scene={
            "robot": SimpleNamespace(data=SimpleNamespace(
                body_pos_w=torch.cat((mount_position, torch.tensor([[[0.5, 0.0, 0.4]]], device=args.device)), dim=1),
                body_quat_w=torch.stack((quaternion, identity), dim=1),
            )),
            "box": SimpleNamespace(data=SimpleNamespace(root_pos_w=box_position, root_quat_w=identity)),
        },
    )
    command = box_push.push_ee_commands(fake, SimpleNamespace(body_ids=[0]))
    recovered = mount_position[:, 0] + quat_apply(quaternion, command[:, :3])
    torch.testing.assert_close(recovered, torch.tensor([[0.50, 0.0, 0.125]], device=args.device))
    fake.scene["robot"].data.body_pos_w[:, 1] = torch.tensor([0.5, 0.0, 0.125], device=args.device)
    command = box_push.push_ee_commands(fake, SimpleNamespace(body_ids=[0]))
    recovered = mount_position[:, 0] + quat_apply(quaternion, command[:, :3])
    torch.testing.assert_close(recovered, torch.tensor([[0.62, 0.0, 0.125]], device=args.device))

    config = load_cfg_from_registry("GO2-ARX5-Box-Push-Play", "env_cfg_entry_point")
    config.scene.num_envs = 4
    config.sim.device = args.device
    env = gym.make("GO2-ARX5-Box-Push-Play", cfg=config).unwrapped
    env.reset()
    mount = SceneEntityCfg("robot", body_names="link0")
    mount.resolve(env.scene)
    hand = SceneEntityCfg("robot", body_names="end_effector")
    hand.resolve(env.scene)
    robot = env.scene["robot"]
    commands = box_push.push_ee_commands(env, mount)
    target = robot.data.body_pos_w[:, mount.body_ids[0]] + quat_apply(robot.data.body_quat_w[:, mount.body_ids[0]], commands[:, :3])
    expected = box_push.front_face(env)
    offset = torch.zeros_like(expected)
    offset[:, 0] = -env.cfg.push_approach_distance
    expected += quat_apply(env.scene["box"].data.root_quat_w, offset)
    torch.testing.assert_close(target, expected)
    assert not box_push.hand_contact(env, hand).any()
    assert not box_push.push_progress_reward(env, hand).any()
    print("PUSH_FRAME_CONTRACT_PASSED", {
        "mount_position": robot.data.body_pos_w[0, mount.body_ids[0]].tolist(),
        "hand_position": robot.data.body_pos_w[0, hand.body_ids[0]].tolist(),
        "face_target": target[0].tolist(), "mount_frame_command": commands[0].tolist(),
    }, flush=True)
    env.close()


try:
    main()
finally:
    launcher.app.close()