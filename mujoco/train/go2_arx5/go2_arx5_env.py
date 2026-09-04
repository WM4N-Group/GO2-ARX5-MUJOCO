"""Gymnasium environment for training GO2-ARX5 directly in MuJoCo."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
import yaml


ROOT_DIR = Path(__file__).resolve().parents[2]
XML_PATH = ROOT_DIR / "robots/go2_arx5/scene.xml"
DEPLOY_CONFIG_PATH = ROOT_DIR / "deploy/deploy_mujoco/go2_arx5/config.yaml"

# Policy order is FR, FL, RR, RL, arm; MuJoCo order is FL, FR, RL, RR, arm.
POLICY_MJ_INDICES = np.array(
    [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8, 12, 13, 14, 15, 16, 17]
)
MJ_POLICY_INDICES = np.argsort(POLICY_MJ_INDICES)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def projected_gravity(quat_wxyz: np.ndarray) -> np.ndarray:
    """Rotate the world gravity direction into the body frame."""
    rotation = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation, quat_wxyz)
    return rotation.reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])


class Go2ARX5Env(gym.Env[np.ndarray, np.ndarray]):
    """GO2-ARX5 flat-ground locomotion and manipulation task."""

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        render_mode: str | None = None,
        episode_length_s: float = 20.0,
        command_profile: str = "train",
    ) -> None:
        super().__init__()
        if render_mode not in (None, "human"):
            raise ValueError(f"Unsupported render mode: {render_mode}")
        if command_profile not in ("train", "full", "stand"):
            raise ValueError(f"Unsupported command profile: {command_profile}")

        self.render_mode = render_mode
        self.command_profile = command_profile
        self.model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        self.data = mujoco.MjData(self.model)
        with DEPLOY_CONFIG_PATH.open(encoding="utf-8") as config_file:
            deploy_cfg = yaml.safe_load(config_file)

        self.sim_dt = float(deploy_cfg["simulation_dt"])
        self.decimation = int(deploy_cfg["control_decimation"])
        self.control_dt = self.sim_dt * self.decimation
        self.model.opt.timestep = self.sim_dt
        self.max_episode_steps = round(episode_length_s / self.control_dt)

        self.default_qpos = np.asarray(deploy_cfg["default_angles"], dtype=np.float64)
        self.kp = np.asarray(deploy_cfg["kps"], dtype=np.float64)
        self.kd = np.asarray(deploy_cfg["kds"], dtype=np.float64)
        self.action_scale = np.asarray(deploy_cfg["action_scale"], dtype=np.float64)
        arrays = (self.default_qpos, self.kp, self.kd, self.action_scale)
        if not all(array.shape == (18,) for array in arrays):
            raise ValueError("GO2-ARX5 deployment config must contain 18 joint values")

        self.ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()
        self.joint_low = self.model.jnt_range[1:, 0].copy()
        self.joint_high = self.model.jnt_range[1:, 1].copy()

        self.base_id = self._body_id("base")
        self.arm_base_id = self._body_id("x5_base_link")
        self.ee_id = self._body_id("x5_link6")
        self.floor_geom_id = self._geom_id("floor")
        self.foot_geom_ids = np.array(
            [self._geom_id(name) for name in ("FR", "FL", "RR", "RL")]
        )

        self.action_space = gym.spaces.Box(
            -10.0, 10.0, shape=(18,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            -100.0, 100.0, shape=(210,), dtype=np.float32
        )

        self.history: deque[np.ndarray] = deque(maxlen=3)
        self.last_action = np.zeros(18, dtype=np.float64)
        self.velocity_command = np.zeros(3, dtype=np.float64)
        self.ee_command = np.array(
            [0.425, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        self.command_steps_left = 0
        self.foot_air_time = np.zeros(4, dtype=np.float64)
        self.previous_contacts = np.zeros(4, dtype=bool)
        self.step_count = 0
        self.viewer = None

    def _body_id(self, name: str) -> int:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"Body not found in GO2-ARX5 model: {name}")
        return body_id

    def _geom_id(self, name: str) -> int:
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id < 0:
            raise ValueError(f"Geom not found in GO2-ARX5 model: {name}")
        return geom_id

    def _sample_command(self) -> None:
        if self.command_profile == "stand":
            self.velocity_command.fill(0.0)
            self.ee_command[:] = [0.425, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0]
        else:
            full = self.command_profile == "full"
            vel_limit = (
                np.array([1.0, 0.6, 1.0])
                if full
                else np.array([0.2, 0.2, 0.2])
            )
            self.velocity_command = self.np_random.uniform(-vel_limit, vel_limit)
            if self.np_random.random() < 0.1:
                self.velocity_command.fill(0.0)
            x_range = (0.45, 0.7) if full else (0.4, 0.45)
            y_range = (-0.35, 0.35) if full else (-0.05, 0.05)
            z_range = (-0.2, 0.5) if full else (0.05, 0.05)
            self.ee_command[:3] = [
                self.np_random.uniform(*x_range),
                self.np_random.uniform(*y_range),
                self.np_random.uniform(*z_range),
            ]
            self.ee_command[3:] = [1.0, 0.0, 0.0, 0.0]
        self.command_steps_left = round(
            self.np_random.uniform(8.0, 10.0) / self.control_dt
        )

    def _base_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.base_id,
            velocity,
            1,
        )
        return velocity[3:].copy(), velocity[:3].copy()

    def _single_observation(self) -> np.ndarray:
        _, angular_velocity = self._base_velocity()
        joint_position = self.data.qpos[7:] - self.default_qpos
        joint_velocity = self.data.qvel[6:]
        observation = np.concatenate(
            [
                angular_velocity * 0.2,
                projected_gravity(self.data.qpos[3:7]),
                joint_position[MJ_POLICY_INDICES],
                joint_velocity[MJ_POLICY_INDICES] * 0.05,
                self.last_action,
                self.velocity_command,
                self.ee_command,
            ]
        )
        return np.clip(observation, -100.0, 100.0).astype(np.float32)

    def _observation(self) -> np.ndarray:
        # Isaac Lab groups the three-frame history by observation term.
        frames = tuple(self.history)
        term_slices = (
            (0, 3),
            (3, 6),
            (6, 24),
            (24, 42),
            (42, 60),
            (60, 63),
            (63, 70),
        )
        return np.concatenate(
            [frame[start:end] for start, end in term_slices for frame in frames]
        ).astype(np.float32)

    def _contacts(self) -> tuple[np.ndarray, bool]:
        foot_contacts = np.zeros(4, dtype=bool)
        base_contact = False
        foot_lookup = {
            geom_id: index for index, geom_id in enumerate(self.foot_geom_ids)
        }
        base_geom_ids = set(
            np.flatnonzero(self.model.geom_bodyid == self.base_id).tolist()
        )
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if contact.geom1 == self.floor_geom_id:
                other = contact.geom2
            elif contact.geom2 == self.floor_geom_id:
                other = contact.geom1
            else:
                continue
            if other in foot_lookup:
                foot_contacts[foot_lookup[other]] = True
            if other in base_geom_ids:
                base_contact = True
        return foot_contacts, base_contact

    def _end_effector_pose(self) -> tuple[np.ndarray, np.ndarray]:
        arm_rotation = self.data.xmat[self.arm_base_id].reshape(3, 3)
        relative_position = arm_rotation.T @ (
            self.data.xpos[self.ee_id] - self.data.xpos[self.arm_base_id]
        )
        relative_quat = quat_multiply(
            quat_conjugate(self.data.xquat[self.base_id]),
            self.data.xquat[self.ee_id],
        )
        return relative_position, relative_quat

    def _reward(
        self, action: np.ndarray
    ) -> tuple[float, dict[str, float], bool]:
        linear_velocity, angular_velocity = self._base_velocity()
        gravity = projected_gravity(self.data.qpos[3:7])
        foot_contacts, base_contact = self._contacts()
        ee_position, ee_quat = self._end_effector_pose()

        first_contact = foot_contacts & ~self.previous_contacts
        landing_air_time = self.foot_air_time.copy()
        self.foot_air_time = np.where(
            foot_contacts, 0.0, self.foot_air_time + self.control_dt
        )
        self.previous_contacts = foot_contacts

        terms: dict[str, float] = {}
        terms["ee_position"] = 3.0 * np.exp(
            -np.sum((ee_position - self.ee_command[:3]) ** 2) / 0.1
        )
        orientation_error = 2.0 * np.arccos(
            np.clip(abs(ee_quat[0]), 0.0, 1.0)
        )
        terms["ee_orientation"] = -1.5 * orientation_error
        terms["linear_velocity"] = 3.0 * np.exp(
            -np.sum((linear_velocity[:2] - self.velocity_command[:2]) ** 2) / 0.25
        )
        terms["yaw_velocity"] = 1.5 * np.exp(
            -((angular_velocity[2] - self.velocity_command[2]) ** 2) / 0.25
        )
        terms["base_height"] = np.exp(
            -abs(self.data.qpos[2] - 0.28) / 0.02
        )
        terms["vertical_velocity"] = -2.0 * linear_velocity[2] ** 2
        terms["roll_pitch_velocity"] = -0.05 * float(
            np.sum(angular_velocity[:2] ** 2)
        )
        terms["flat_orientation"] = -float(np.sum(gravity[:2] ** 2))
        terms["torque"] = -1.0e-5 * float(np.sum(self.data.ctrl**2))
        terms["joint_acceleration"] = -2.5e-7 * float(
            np.sum(self.data.qacc[6:] ** 2)
        )
        terms["action_rate"] = -0.01 * float(
            np.sum((action - self.last_action) ** 2)
        )
        terms["joint_power"] = -2.0e-5 * float(
            np.sum(np.abs(self.data.ctrl * self.data.qvel[6:]))
        )

        slide = 0.0
        for foot_index, geom_id in enumerate(self.foot_geom_ids):
            if foot_contacts[foot_index]:
                velocity = np.zeros(6, dtype=np.float64)
                mujoco.mj_objectVelocity(
                    self.model,
                    self.data,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    int(geom_id),
                    velocity,
                    0,
                )
                slide += np.linalg.norm(velocity[3:5])
        upright_scale = np.clip(-gravity[2], 0.0, 0.7) / 0.7
        terms["feet_slide"] = -0.1 * slide * upright_scale
        if np.linalg.norm(self.velocity_command[:2]) > 0.1:
            terms["feet_air_time"] = 0.5 * float(
                np.sum((landing_air_time - 0.5) * first_contact)
            )
        else:
            terms["feet_air_time"] = 0.0
        terms["feet_long_air"] = -0.5 * float(
            np.sum(np.maximum(self.foot_air_time - 1.0, 0.0) ** 2)
        )
        terms["air_time_variance"] = -float(
            np.var(np.minimum(self.foot_air_time, 0.5))
        )

        joint_delta = self.data.qpos[7:] - self.default_qpos
        terms["joint_deviation"] = -upright_scale * (
            0.1 * float(np.sum(np.abs(joint_delta[[0, 3, 6, 9]])))
            + 0.02
            * float(
                np.sum(np.abs(joint_delta[[1, 2, 4, 5, 7, 8, 10, 11]]))
            )
            + 0.1 * float(np.sum(np.abs(joint_delta[12:])))
        )
        policy_joint_pos = self.data.qpos[7:][MJ_POLICY_INDICES]
        mirror_error = np.sum(
            (policy_joint_pos[0:3] - policy_joint_pos[6:9]) ** 2
        )
        mirror_error += np.sum(
            (policy_joint_pos[3:6] - policy_joint_pos[9:12]) ** 2
        )
        terms["joint_mirror"] = -0.075 * mirror_error * upright_scale
        lower_violation = np.maximum(
            self.joint_low - self.data.qpos[7:], 0.0
        )
        upper_violation = np.maximum(
            self.data.qpos[7:] - self.joint_high, 0.0
        )
        terms["joint_limits"] = -float(
            np.sum(lower_violation + upper_violation)
        )

        bad_orientation = -gravity[2] < np.cos(1.0)
        finite_state = np.isfinite(self.data.qpos).all() and np.isfinite(
            self.data.qvel
        ).all()
        terminated = bool(base_contact or bad_orientation or not finite_state)
        if terminated:
            terms["termination"] = -5.0

        scaled_terms = {
            name: float(value * self.control_dt) for name, value in terms.items()
        }
        return float(sum(scaled_terms.values())), scaled_terms, terminated

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[7:] = self.default_qpos + self.np_random.uniform(
            -0.02, 0.02, 18
        )
        yaw = self.np_random.uniform(-np.pi, np.pi)
        self.data.qpos[3:7] = [
            np.cos(yaw / 2.0),
            0.0,
            0.0,
            np.sin(yaw / 2.0),
        ]
        self.data.qvel[:] = self.np_random.uniform(
            -0.01, 0.01, self.model.nv
        )
        self.last_action.fill(0.0)
        self.foot_air_time.fill(0.0)
        self.previous_contacts.fill(False)
        self.step_count = 0
        self._sample_command()
        mujoco.mj_forward(self.model, self.data)

        frame = self._single_observation()
        self.history.clear()
        self.history.extend(frame.copy() for _ in range(3))
        if self.render_mode == "human":
            self.render()
        return self._observation(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(np.asarray(action, dtype=np.float64), -10.0, 10.0)
        target_policy = (
            self.default_qpos[MJ_POLICY_INDICES]
            + action * self.action_scale[MJ_POLICY_INDICES]
        )
        target_mj = target_policy[POLICY_MJ_INDICES]
        target_mj = np.clip(target_mj, self.joint_low, self.joint_high)

        for _ in range(self.decimation):
            torque = self.kp * (target_mj - self.data.qpos[7:])
            torque -= self.kd * self.data.qvel[6:]
            self.data.ctrl[:] = np.clip(
                torque, self.ctrl_low, self.ctrl_high
            )
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        self.command_steps_left -= 1
        if self.command_steps_left <= 0:
            self._sample_command()

        reward, reward_terms, terminated = self._reward(action)
        self.last_action = action.copy()
        self.history.append(self._single_observation())
        truncated = self.step_count >= self.max_episode_steps
        info: dict[str, Any] = {"reward_terms": reward_terms}
        info.update(
            {f"reward/{name}": value for name, value in reward_terms.items()}
        )
        if self.render_mode == "human":
            self.render()
        return self._observation(), reward, terminated, truncated, info

    def render(self) -> None:
        if self.render_mode != "human":
            return
        if self.viewer is None:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.viewer.cam.trackbodyid = self.base_id
            self.viewer.cam.distance = 3.0
            self.viewer.cam.elevation = -20
        if self.viewer.is_running():
            self.viewer.sync()

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
