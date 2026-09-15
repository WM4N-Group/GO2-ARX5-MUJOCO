"""TorchScript locomotion-policy runtime for reconfigurable MuJoCo scenes."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from pathlib import Path

import mujoco
import numpy as np
import torch

from .env import BlockedPassageEnv


ROOT_DIR = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT_DIR / "deploy/policy/go2_arx5/policy.pt"

# Policy order is FR, FL, RR, RL, arm; MuJoCo order is FL, FR, RL, RR, arm.
POLICY_MJ_INDICES = np.array(
    [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8, 12, 13, 14, 15, 16, 17]
)
MJ_POLICY_INDICES = np.argsort(POLICY_MJ_INDICES)
JOINT_NAMES = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "x5_joint1",
    "x5_joint2",
    "x5_joint3",
    "x5_joint4",
    "x5_joint5",
    "x5_joint6",
)


def projected_gravity(quat_wxyz: np.ndarray) -> np.ndarray:
    rotation = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation, quat_wxyz)
    return rotation.reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])


class LocomotionRuntime:
    """Run the existing 210-to-18 policy without assuming a fixed model nq."""

    def __init__(
        self,
        env: BlockedPassageEnv,
        policy_path: Path | str = POLICY_PATH,
        action_clip: float = 20.0,
        *,
        joint_names: tuple[str, ...] = JOINT_NAMES,
        base_body_name: str = "base",
        on_physics_step: Callable[[], None] | None = None,
    ) -> None:
        if action_clip <= 0.0:
            raise ValueError("action_clip must be positive")
        if len(joint_names) != 18 or len(set(joint_names)) != 18:
            raise ValueError("Expected 18 unique robot joint names")
        self.action_clip = action_clip
        self.on_physics_step = on_physics_step
        self.env = env
        self.model = env.model
        self.data = env.data
        cfg = env.deploy_cfg
        self.sim_dt = float(cfg["simulation_dt"])
        self.decimation = int(cfg["control_decimation"])
        self.control_dt = self.sim_dt * self.decimation
        self.model.opt.timestep = self.sim_dt

        self.default_qpos = np.asarray(cfg["default_angles"], dtype=np.float64)
        self.kp = np.asarray(cfg["kps"], dtype=np.float64)
        self.kd = np.asarray(cfg["kds"], dtype=np.float64)
        self.action_scale = np.asarray(cfg["action_scale"], dtype=np.float64)
        self.ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()

        joint_ids = np.array([self._joint_id(name) for name in joint_names])
        self.joint_qpos_adr = self.model.jnt_qposadr[joint_ids].astype(int)
        self.joint_dof_adr = self.model.jnt_dofadr[joint_ids].astype(int)
        self.joint_low = self.model.jnt_range[joint_ids, 0].copy()
        self.joint_high = self.model.jnt_range[joint_ids, 1].copy()
        if self.model.nu != 18:
            raise ValueError(f"Expected 18 actuators, got {self.model.nu}")

        self.base_id = self._body_id(base_body_name)
        self.robot_dof_adr = int(
            self.model.jnt_dofadr[self.env.robot_joint_id]
        )
        self.policy = torch.jit.load(str(policy_path), map_location="cpu")
        self.policy.eval()
        self.history: deque[np.ndarray] = deque(maxlen=3)
        self.last_action = np.zeros(18, dtype=np.float64)
        self.velocity_command = np.zeros(3, dtype=np.float64)
        self.ee_command = np.array(
            [0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        self.reset()

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

    def reset(self) -> None:
        self.last_action.fill(0.0)
        self.velocity_command.fill(0.0)
        frame = self._single_observation()
        self.history.clear()
        self.history.extend(frame.copy() for _ in range(3))

    def _base_angular_velocity(self) -> np.ndarray:
        start = self.robot_dof_adr + 3
        return self.data.qvel[start : start + 3].copy()

    def _single_observation(self) -> np.ndarray:
        joint_position = self.data.qpos[self.joint_qpos_adr] - self.default_qpos
        joint_velocity = self.data.qvel[self.joint_dof_adr]
        observation = np.concatenate(
            [
                self._base_angular_velocity() * 0.2,
                projected_gravity(
                    self.data.qpos[
                        self.env.robot_qpos_adr + 3 : self.env.robot_qpos_adr + 7
                    ]
                ),
                joint_position[MJ_POLICY_INDICES],
                joint_velocity[MJ_POLICY_INDICES] * 0.05,
                self.last_action,
                self.velocity_command,
                self.ee_command,
            ]
        )
        return np.clip(observation, -100.0, 100.0).astype(np.float32)

    def observation(self) -> np.ndarray:
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

    def hold_default(self) -> None:
        """Advance one control period while holding the nominal joint pose."""
        self._simulate_target(self.default_qpos)
        self.last_action.fill(0.0)
        self.history.append(self._single_observation())

    def _simulate_target(self, target_mj: np.ndarray) -> None:
        for _ in range(self.decimation):
            joint_position = self.data.qpos[self.joint_qpos_adr]
            joint_velocity = self.data.qvel[self.joint_dof_adr]
            torque = self.kp * (target_mj - joint_position)
            torque -= self.kd * joint_velocity
            self.data.ctrl[:] = np.clip(torque, self.ctrl_low, self.ctrl_high)
            mujoco.mj_step(self.model, self.data)
            if self.on_physics_step is not None:
                self.on_physics_step()

    def step(
        self,
        velocity_command: np.ndarray,
        ee_command: np.ndarray | None = None,
    ) -> np.ndarray:
        velocity = np.asarray(velocity_command, dtype=np.float64)
        if velocity.shape != (3,):
            raise ValueError("velocity_command must have shape (3,)")
        self.velocity_command[:] = velocity
        if ee_command is not None:
            ee_pose = np.asarray(ee_command, dtype=np.float64)
            if ee_pose.shape != (7,):
                raise ValueError("ee_command must have shape (7,)")
            self.ee_command[:] = ee_pose

        policy_observation = torch.from_numpy(self.observation()).unsqueeze(0)
        with torch.inference_mode():
            action = self.policy(policy_observation).squeeze(0).cpu().numpy()
        action = np.clip(
            action, -self.action_clip, self.action_clip
        ).astype(np.float64)
        target_policy = (
            self.default_qpos[MJ_POLICY_INDICES]
            + action * self.action_scale[MJ_POLICY_INDICES]
        )
        target_mj = np.clip(
            target_policy[POLICY_MJ_INDICES], self.joint_low, self.joint_high
        )

        self._simulate_target(target_mj)

        self.last_action = action
        self.history.append(self._single_observation())
        return action.astype(np.float32)
