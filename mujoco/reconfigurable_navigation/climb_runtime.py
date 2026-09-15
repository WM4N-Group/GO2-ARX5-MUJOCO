"""TorchScript CLIMB-policy runtime for MuJoCo skill switching."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Mapping

import mujoco
import numpy as np
import torch
import yaml

from .locomotion_runtime import (
    JOINT_NAMES,
    MJ_POLICY_INDICES,
    POLICY_MJ_INDICES,
    projected_gravity,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
XML_PATH = ROOT_DIR / "train/go2_arx5/climb_scene.xml"
CONFIG_PATH = ROOT_DIR / "deploy/deploy_mujoco/go2_arx5/config.yaml"
POLICY_PATH = ROOT_DIR / "deploy/policy/go2_arx5/climb/policy_iter1499.pt"
HEIGHT_SCAN_SIZE = 187
HEIGHT_SCAN_OFFSET = 0.5
ACTUATOR_DELAY_STEPS = np.array(
    [4] * 12 + [3, 0, 4, 0, 0, 0], dtype=np.int64
)
JOINT_VELOCITY_LIMITS = np.array(
    [30.1, 30.1, 15.7] * 4 + [1000.0] * 6, dtype=np.float64
)


class ClimbRuntime:
    """Run the 253-to-18 CLIMB actor on owned or shared MuJoCo state."""

    def __init__(
        self,
        policy_path: Path | str = POLICY_PATH,
        *,
        model: mujoco.MjModel | None = None,
        data: mujoco.MjData | None = None,
        deploy_config: Mapping[str, Any] | None = None,
        joint_names: tuple[str, ...] = JOINT_NAMES,
        base_body_name: str = "base",
        linear_velocity_at_com: bool = False,
    ) -> None:
        if (model is None) != (data is None):
            raise ValueError("model and data must be provided together")
        self.owns_physics = model is None
        self.linear_velocity_at_com = linear_velocity_at_com
        self.model = model or mujoco.MjModel.from_xml_path(str(XML_PATH))
        self.data = data or mujoco.MjData(self.model)

        if deploy_config is None:
            with CONFIG_PATH.open(encoding="utf-8") as config_file:
                config = yaml.safe_load(config_file)
        else:
            config = deploy_config
        self.sim_dt = float(config["simulation_dt"])
        self.decimation = int(config["control_decimation"])
        self.control_dt = self.sim_dt * self.decimation
        self.model.opt.timestep = self.sim_dt
        self.default_qpos = np.asarray(config["default_angles"], dtype=np.float64)
        self.kp = np.asarray(config["kps"], dtype=np.float64).copy()
        self.kd = np.asarray(config["kds"], dtype=np.float64).copy()
        self.kp[12:] = [50.0, 50.0, 80.0, 30.0, 30.0, 20.0]
        self.kd[12:] = [3.0, 2.0, 3.0, 3.0, 2.5, 1.0]
        self.action_scale = np.asarray(config["action_scale"], dtype=np.float64)
        self.ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()

        if len(joint_names) != 18 or len(set(joint_names)) != 18:
            raise ValueError("Expected 18 unique joint names")
        joint_ids = np.array([self._joint_id(name) for name in joint_names])
        self.joint_qpos_adr = self.model.jnt_qposadr[joint_ids].astype(int)
        self.joint_dof_adr = self.model.jnt_dofadr[joint_ids].astype(int)
        self.base_id = self._body_id(base_body_name)
        self.base_geom_ids = frozenset(
            index
            for index, body_id in enumerate(self.model.geom_bodyid)
            if int(body_id) == self.base_id
            and self.model.geom_contype[index] != 0
        )
        self.terrain_geom_ids = frozenset(
            index
            for index, body_id in enumerate(self.model.geom_bodyid)
            if int(body_id) == 0 and self.model.geom_contype[index] != 0
        )

        self.policy = torch.jit.load(str(policy_path), map_location="cpu")
        self.policy.eval()
        self.actuator_delay_steps = ACTUATOR_DELAY_STEPS.copy()
        self.joint_velocity_limits = JOINT_VELOCITY_LIMITS.copy()
        self.target_history: deque[np.ndarray] = deque(
            maxlen=int(self.actuator_delay_steps.max()) + 1
        )
        self.last_action = np.zeros(18, dtype=np.float64)
        self.velocity_command = np.array([0.5, 0.0, 0.0], dtype=np.float64)
        x = np.arange(-0.8, 0.8 + 1.0e-9, 0.1)
        y = np.arange(-0.5, 0.5 + 1.0e-9, 0.1)
        grid_x, grid_y = np.meshgrid(x, y, indexing="xy")
        self.height_scan_points = np.column_stack(
            (grid_x.ravel() + 0.2, grid_y.ravel())
        )
        self.terrain_geom_group = np.array(
            [1, 0, 0, 0, 0, 0], dtype=np.uint8
        )
        if self.height_scan_points.shape != (HEIGHT_SCAN_SIZE, 2):
            raise ValueError("Height scan grid must contain 187 points")

        if self.owns_physics:
            self.reset()
        else:
            self.activate()

    def _joint_id(self, name: str) -> int:
        joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        if joint_id < 0:
            raise ValueError(f"Joint not found: {name}")
        return joint_id

    def _body_id(self, name: str) -> int:
        body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, name
        )
        if body_id < 0:
            raise ValueError(f"Body not found: {name}")
        return body_id

    def activate(self) -> None:
        """Reset policy memory without modifying the shared physical state."""
        self.last_action.fill(0.0)
        self.target_history.clear()
        self.target_history.extend(
            np.zeros(18, dtype=np.float64)
            for _ in range(self.target_history.maxlen)
        )

    def reset(self) -> None:
        """Reset an owned CLIMB scene and initialize policy memory."""
        if not self.owns_physics:
            raise RuntimeError("Cannot reset shared MuJoCo physics")
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 0.55
        self.data.qpos[self.joint_qpos_adr] = self.default_qpos
        self.data.qpos[self.joint_qpos_adr[13:15]] = 0.15
        self.data.qvel.fill(0.0)
        self.activate()
        mujoco.mj_forward(self.model, self.data)

    def _base_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        rotation = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rotation, self.data.qpos[3:7])
        linear_velocity = rotation.reshape(3, 3).T @ self.data.qvel[:3]
        if self.linear_velocity_at_com:
            linear_velocity += np.cross(self.data.qvel[3:6], self.model.body_ipos[self.base_id])
        return linear_velocity, self.data.qvel[3:6].copy()

    def observation(self) -> np.ndarray:
        linear_velocity, angular_velocity = self._base_velocity()
        joint_position = self.data.qpos[self.joint_qpos_adr] - self.default_qpos
        joint_velocity = self.data.qvel[self.joint_dof_adr]
        observation = np.concatenate(
            [
                linear_velocity,
                angular_velocity,
                projected_gravity(self.data.xquat[self.base_id]),
                self.velocity_command,
                joint_position[MJ_POLICY_INDICES],
                joint_velocity[MJ_POLICY_INDICES],
                self.last_action,
                self._height_scan(),
            ]
        )
        if observation.shape != (253,):
            raise ValueError(f"Expected 253 observations, got {observation.shape}")
        return observation.astype(np.float32)

    def _height_scan(self) -> np.ndarray:
        base_position = self.data.xpos[self.base_id]
        base_quaternion = self.data.xquat[self.base_id]
        yaw = np.arctan2(
            2.0
            * (
                base_quaternion[0] * base_quaternion[3]
                + base_quaternion[1] * base_quaternion[2]
            ),
            1.0
            - 2.0
            * (base_quaternion[2] ** 2 + base_quaternion[3] ** 2),
        )
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        height_scan = np.empty(HEIGHT_SCAN_SIZE, dtype=np.float64)
        direction = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        for index, point in enumerate(self.height_scan_points):
            origin = np.array(
                [
                    base_position[0] + cos_yaw * point[0] - sin_yaw * point[1],
                    base_position[1] + sin_yaw * point[0] + cos_yaw * point[1],
                    base_position[2] + 20.0,
                ],
                dtype=np.float64,
            )
            distance = mujoco.mj_ray(
                self.model,
                self.data,
                origin,
                direction,
                self.terrain_geom_group,
                True,
                -1,
                None,
            )
            hit_height = origin[2] - distance if distance >= 0.0 else 0.0
            height_scan[index] = (
                base_position[2] - hit_height - HEIGHT_SCAN_OFFSET
            )
        return np.clip(height_scan, -1.0, 1.0)

    def hold_default(self) -> None:
        self._simulate_target(self.default_qpos)
        self.last_action.fill(0.0)

    def step(self, velocity_command: np.ndarray | None = None) -> np.ndarray:
        if velocity_command is not None:
            velocity = np.asarray(velocity_command, dtype=np.float64)
            if velocity.shape != (3,):
                raise ValueError("velocity_command must have shape (3,)")
            self.velocity_command[:] = velocity
        policy_observation = torch.from_numpy(self.observation()).unsqueeze(0)
        with torch.inference_mode():
            action = self.policy(policy_observation).squeeze(0).cpu().numpy()
        action = np.clip(action, -20.0, 20.0).astype(np.float64)
        target_policy = (
            self.default_qpos[MJ_POLICY_INDICES]
            + action * self.action_scale[MJ_POLICY_INDICES]
        )
        self._simulate_target(target_policy[POLICY_MJ_INDICES])
        self.last_action = action
        return action.astype(np.float32)

    def _simulate_target(self, target: np.ndarray) -> None:
        for _ in range(self.decimation):
            self.target_history.append(target.copy())
            target_history = tuple(self.target_history)
            delayed_target = np.array(
                [
                    target_history[-1 - delay][joint_index]
                    for joint_index, delay in enumerate(self.actuator_delay_steps)
                ]
            )
            joint_position = self.data.qpos[self.joint_qpos_adr]
            joint_velocity = self.data.qvel[self.joint_dof_adr]
            torque = (
                self.kp * (delayed_target - joint_position)
                - self.kd * joint_velocity
            )
            self.data.ctrl[:] = np.clip(torque, self.ctrl_low, self.ctrl_high)
            previous_joint_position = joint_position.copy()
            mujoco.mj_step(self.model, self.data)
            limited_joint_velocity = np.clip(
                self.data.qvel[self.joint_dof_adr],
                -self.joint_velocity_limits,
                self.joint_velocity_limits,
            )
            self.data.qvel[self.joint_dof_adr] = limited_joint_velocity
            self.data.qpos[self.joint_qpos_adr] = (
                previous_joint_position + limited_joint_velocity * self.sim_dt
            )
            mujoco.mj_forward(self.model, self.data)
            if getattr(self, "on_physics_step", None) is not None:
                self.on_physics_step()

    def base_contacts_terrain(self) -> bool:
        for contact in self.data.contact[: self.data.ncon]:
            pair = {int(contact.geom1), int(contact.geom2)}
            if pair & self.base_geom_ids and pair & self.terrain_geom_ids:
                return True
        return False