"""Goal and support conditions for ground-to-box climbing."""

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat
from .box_rewards import directed_progress, support_clearance, top_surface_support, valid_skill_success, velocity_tracking_advantage


def box_target_delta(env):
    robot = env.scene["robot"]
    config = env.cfg.scene.terrain.terrain_generator.sub_terrains["box"]
    target = env.scene.env_origins.clone()
    target[:, 0] += config.approach_distance + config.box_length / 2.0
    target[:, 2] = robot.data.root_pos_w[:, 2]
    return quat_apply_inverse(yaw_quat(robot.data.root_quat_w), target - robot.data.root_pos_w)


def box_velocity_commands(env):
    delta = box_target_delta(env)
    commands = torch.zeros_like(delta)
    commands[:, 0] = torch.clamp(0.8 * delta[:, 0], -0.15, 0.30)
    commands[:, 1] = torch.clamp(0.8 * delta[:, 1], -0.12, 0.12)
    commands[torch.linalg.vector_norm(delta[:, :2], dim=1) < 0.10] = 0.0
    return commands


def track_box_velocity(env, std: float = 0.4):
    return velocity_tracking_advantage(
        box_velocity_commands(env)[:, :2], env.scene["robot"].data.root_lin_vel_b[:, :2], std,
    )


def box_goal_reward(env):
    delta = box_target_delta(env)[:, :2]
    return directed_progress(
        torch.zeros_like(delta), delta, env.scene["robot"].data.root_lin_vel_b[:, :2],
        torch.ones(env.num_envs, device=env.device, dtype=torch.bool),
    )


def supported_feet(env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg):
    robot = env.scene[asset_cfg.name]
    feet = robot.data.body_pos_w[:, asset_cfg.body_ids] - env.scene.env_origins[:, None, :]
    config = env.cfg.scene.terrain.terrain_generator.sub_terrains["box"]
    inside = (
        (feet[:, :, 0] > config.approach_distance + 0.04)
        & (feet[:, :, 0] < config.approach_distance + config.box_length - 0.04)
        & (feet[:, :, 1].abs() < config.box_width / 2.0 - 0.04)
    )
    forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids, 2]
    hits = env.scene["height_scanner"].data.ray_hits_w - env.scene.env_origins[:, None, :]
    target_hits = (
        torch.isfinite(hits).all(dim=-1)
        & (hits[..., 0] > config.approach_distance)
        & (hits[..., 0] < config.approach_distance + config.box_length)
        & (hits[..., 1].abs() < config.box_width / 2.0)
    )
    height = torch.where(target_hits, hits[..., 2], -torch.inf).max(dim=1).values
    return inside & top_surface_support(feet[:, :, 2], forces, height)


def box_support_reward(env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg):
    return supported_feet(env, asset_cfg, sensor_cfg).float().mean(dim=1)


def box_clearance_reward(env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg):
    robot = env.scene[asset_cfg.name]
    feet = robot.data.body_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2, None]
    forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids, 2]
    contacts = (forces > 2.0).float()
    support_height = ((feet - 0.0225) * contacts).sum(dim=1) / contacts.sum(dim=1).clamp_min(1.0)
    base_height = robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    error = base_height - support_height - 0.30
    return torch.clamp(1.0 - error.abs() / 0.20, min=0.0) * (contacts.sum(dim=1) >= 2)


class LowPosture(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.duration = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None):
        self.duration[slice(None) if env_ids is None else env_ids] = 0.0

    def __call__(self, env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, min_clearance=0.18, grace_time=0.4, duration=0.2):
        robot = env.scene[asset_cfg.name]
        clearance = support_clearance(
            robot.data.root_pos_w[:, 2], robot.data.body_pos_w[:, asset_cfg.body_ids, 2],
            env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids, 2],
        )
        low = (clearance < min_clearance) & (env.episode_length_buf * env.step_dt >= grace_time)
        self.duration = torch.where(low, self.duration + env.step_dt, 0.0)
        return self.duration >= duration


def posture_failure_cost(env):
    failed = env.termination_manager.get_term("base_contact") | env.termination_manager.get_term("bad_orientation") | env.termination_manager.get_term("low_posture")
    return failed.float() / env.step_dt


def box_standing(env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg):
    robot = env.scene[asset_cfg.name]
    return (
        supported_feet(env, asset_cfg, sensor_cfg).all(dim=1)
        & (torch.linalg.vector_norm(box_target_delta(env)[:, :2], dim=1) < 0.20)
        & (torch.linalg.vector_norm(robot.data.root_lin_vel_w, dim=1) < 0.15)
        & (torch.linalg.vector_norm(robot.data.root_ang_vel_w, dim=1) < 0.4)
        & (robot.data.projected_gravity_b[:, 2] < -0.94)
    )


class BoxSettled(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.duration = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None):
        self.duration[slice(None) if env_ids is None else env_ids] = 0.0

    def __call__(self, env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, hold_time: float = 1.0):
        stable = box_standing(env, asset_cfg, sensor_cfg)
        self.duration = torch.where(stable, self.duration + env.step_dt, 0.0)
        return self.duration >= hold_time


def box_terrain_levels(env, env_ids):
    terrain = env.scene.terrain
    if env.common_step_counter > 0:
        success = valid_skill_success(env, ("base_contact", "bad_orientation", "low_posture"))[env_ids]
        terrain.update_env_origins(env_ids, success, ~success)
    return terrain.terrain_levels.float().mean()


def feet_flight(env, sensor_cfg: SceneEntityCfg):
    forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids]
    return (torch.linalg.vector_norm(forces, dim=-1).max(dim=1).values < 2.0).float()