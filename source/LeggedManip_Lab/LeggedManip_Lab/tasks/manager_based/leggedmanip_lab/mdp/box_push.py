"""Physical arm-pushing commands, observations and termination conditions."""

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_euler_xyz, yaw_quat
from .box_rewards import directed_progress, velocity_tracking_advantage
from .box_push_control import contact_push_velocity, side_approach_ready, valid_front_contact


NON_HAND_BODIES = ("base",) + tuple(
    f"{leg}_{part}" for leg in ("FL", "FR", "RL", "RR") for part in ("hip", "thigh", "calf", "foot")
) + tuple(f"link{index}" for index in range(6))


def push_target(env):
    if not hasattr(env, "box_push_target"):
        env.box_push_target = env.scene["box"].data.default_root_state[:, :3].clone() + env.scene.env_origins
        env.box_push_target[:, 0] += env.cfg.push_distance
    return env.box_push_target


def reset_push_box(env, env_ids, asset_cfg=SceneEntityCfg("box")):
    box = env.scene[asset_cfg.name]
    state = box.data.default_root_state[env_ids].clone()
    state[:, :3] += env.scene.env_origins[env_ids]
    state[:, :2] += torch.empty((len(env_ids), 2), device=env.device).uniform_(-0.03, 0.03)
    yaw = torch.empty(len(env_ids), device=env.device).uniform_(-0.04, 0.04)
    state[:, 3:7] = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    state[:, 7:] = 0.0
    box.write_root_pose_to_sim(state[:, :7], env_ids)
    box.write_root_velocity_to_sim(state[:, 7:], env_ids)
    target = push_target(env)
    target[env_ids] = state[:, :3]
    target[env_ids, 0] += env.cfg.push_distance
    if hasattr(env, "box_push_approached"):
        env.box_push_approached[env_ids] = False
    if hasattr(env, "box_push_pose_override_active"):
        env.box_push_pose_override_active[env_ids] = False


def front_face(env):
    box = env.scene["box"]
    offset = torch.zeros((env.num_envs, 3), device=env.device)
    offset[:, 0] = -env.cfg.box_size[0] / 2.0
    return box.data.root_pos_w + quat_apply(box.data.root_quat_w, offset)


def push_remaining_distance(env):
    return push_target(env)[:, 0] - env.scene["box"].data.root_pos_w[:, 0]


def push_velocity_commands(env, asset_cfg=SceneEntityCfg("robot", body_names="end_effector")):
    robot, box = env.scene["robot"], env.scene["box"]
    rotation = yaw_quat(robot.data.root_quat_w)
    face = quat_apply_inverse(rotation, front_face(env) - robot.data.root_pos_w)
    box_velocity = quat_apply_inverse(rotation, box.data.root_lin_vel_w)
    distance = push_remaining_distance(env)
    direction = quat_apply_inverse(rotation, push_target(env) - box.data.root_pos_w)
    heading_error = torch.atan2(direction[:, 1], direction[:, 0])
    valid_contact = hand_contact(env, asset_cfg) & ~forbidden_box_contact(env)
    return contact_push_velocity(face, box_velocity, distance, valid_contact, heading_error)


def push_hand_target(env):
    box = env.scene["box"]
    if not hasattr(env, "box_push_hand_id"):
        env.box_push_hand_id = env.scene["robot"].find_bodies("end_effector")[0][0]
    if not hasattr(env, "box_push_approached"):
        env.box_push_approached = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    hand = env.scene["robot"].data.body_pos_w[:, env.box_push_hand_id]
    local = quat_apply_inverse(box.data.root_quat_w, hand - box.data.root_pos_w)
    ready = side_approach_ready(local, env.cfg.box_size[0] / 2.0, env.cfg.push_approach_distance)
    env.box_push_approached |= ready
    env.box_push_approached &= local[:, 2].abs() < 0.06
    at_goal = push_remaining_distance(env) < 0.10
    offset = torch.zeros_like(box.data.root_pos_w)
    offset[:, 0] = -env.cfg.box_size[0] / 2.0 + torch.where(
        env.box_push_approached & ~at_goal, env.cfg.push_press_depth, -env.cfg.push_approach_distance,
    )
    return box.data.root_pos_w + quat_apply(box.data.root_quat_w, offset)


def push_ee_commands(env, asset_cfg=SceneEntityCfg("robot", body_names="link0")):
    robot = env.scene["robot"]
    target = push_hand_target(env)
    mount = asset_cfg.body_ids[0]
    position = quat_apply_inverse(robot.data.body_quat_w[:, mount], target - robot.data.body_pos_w[:, mount])
    orientation = torch.zeros((env.num_envs, 4), device=env.device)
    orientation[:, 0] = 1.0
    command = torch.cat((position, orientation), dim=1)
    if hasattr(env, "box_push_pose_override_active"):
        command = torch.where(env.box_push_pose_override_active[:, None], env.box_push_pose_override, command)
    return command


def box_observation(env):
    robot, box = env.scene["robot"], env.scene["box"]
    rotation = yaw_quat(robot.data.root_quat_w)
    position = quat_apply_inverse(rotation, box.data.root_pos_w - robot.data.root_pos_w)
    velocity = quat_apply_inverse(rotation, box.data.root_lin_vel_w)
    target = quat_apply_inverse(rotation, push_target(env) - box.data.root_pos_w)
    direction = torch.zeros_like(position)
    direction[:, 0] = 1.0
    heading = quat_apply_inverse(rotation, quat_apply(box.data.root_quat_w, direction))[:, :2]
    size = torch.tensor(env.cfg.box_size, device=env.device).expand(env.num_envs, -1)
    return torch.cat((position, velocity, target, heading, size), dim=1)


def hand_contact(env, asset_cfg=SceneEntityCfg("robot", body_names="end_effector")):
    sensor = env.scene.sensors["box_contact_hand"]
    force = sensor.data.force_matrix_w.sum(dim=(1, 2))
    contact = sensor.data.contact_pos_w[:, 0, 0]
    finite = torch.isfinite(contact).all(dim=1)
    hand = env.scene["robot"].data.body_pos_w[:, asset_cfg.body_ids[0]]
    near_tip = torch.linalg.vector_norm(torch.nan_to_num(contact) - hand, dim=1) < 0.07
    box = env.scene["box"]
    local = quat_apply_inverse(box.data.root_quat_w, contact - box.data.root_pos_w)
    axis = torch.zeros_like(force)
    axis[:, 0] = 1.0
    axis = quat_apply(box.data.root_quat_w, axis)
    return valid_front_contact(force, axis, local, near_tip & finite, env.cfg.box_size)


def invalid_hand_contact(env, asset_cfg=SceneEntityCfg("robot", body_names="end_effector")):
    force = env.scene.sensors["box_contact_hand"].data.force_matrix_w.sum(dim=(1, 2))
    return (torch.linalg.vector_norm(force, dim=1) > 1.0) & ~hand_contact(env, asset_cfg)


def forbidden_box_contact(env):
    contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name in NON_HAND_BODIES:
        forces = env.scene.sensors[f"box_contact_{name}"].data.force_matrix_w
        contact |= torch.linalg.vector_norm(forces.sum(dim=(1, 2)), dim=1) > 1.0
    return contact


def box_tipped(env):
    quaternion = env.scene["box"].data.root_quat_w
    return 1.0 - 2.0 * (quaternion[:, 1].square() + quaternion[:, 2].square()) < 0.94


def push_goal_reward(env):
    distance = torch.linalg.vector_norm(push_target(env)[:, :2] - env.scene["box"].data.root_pos_w[:, :2], dim=1)
    return 1.0 - torch.clamp(distance / env.cfg.push_distance, 0.0, 1.0)


def push_progress_reward(env, asset_cfg=SceneEntityCfg("robot", body_names="end_effector")):
    box = env.scene["box"]
    return directed_progress(
        box.data.root_pos_w[:, :2], push_target(env)[:, :2], box.data.root_lin_vel_w[:, :2],
        hand_contact(env, asset_cfg) & ~forbidden_box_contact(env),
    )


def push_hand_tracking(env, asset_cfg=SceneEntityCfg("robot", body_names="end_effector")):
    robot = env.scene["robot"]
    target = push_hand_target(env)
    error = target - robot.data.body_pos_w[:, asset_cfg.body_ids[0]]
    return torch.exp(-torch.linalg.vector_norm(error, dim=1) / 0.20)


def track_push_velocity(env, std=0.35, asset_cfg=SceneEntityCfg("robot", body_names="end_effector")):
    return velocity_tracking_advantage(
        push_velocity_commands(env, asset_cfg)[:, :2], env.scene["robot"].data.root_lin_vel_b[:, :2], std,
    )


def track_push_yaw(env, std=0.5, asset_cfg=SceneEntityCfg("robot", body_names="end_effector")):
    error = push_velocity_commands(env, asset_cfg)[:, 2] - env.scene["robot"].data.root_ang_vel_b[:, 2]
    return torch.exp(-error.square() / std**2)


class PushSettled(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.duration = torch.zeros(env.num_envs, device=env.device)
        self.contact_seen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        selected = slice(None) if env_ids is None else env_ids
        self.duration[selected] = 0.0
        self.contact_seen[selected] = False

    def __call__(self, env, asset_cfg=SceneEntityCfg("robot", body_names="end_effector"), hold_time=0.5):
        self.contact_seen |= hand_contact(env, asset_cfg)
        box = env.scene["box"]
        settled = (
            (torch.linalg.vector_norm(push_target(env)[:, :2] - box.data.root_pos_w[:, :2], dim=1) < 0.12)
            & (torch.linalg.vector_norm(box.data.root_lin_vel_w, dim=1) < 0.08)
            & self.contact_seen & ~box_tipped(env) & ~forbidden_box_contact(env)
        )
        self.duration = torch.where(settled, self.duration + env.step_dt, 0.0)
        return self.duration >= hold_time