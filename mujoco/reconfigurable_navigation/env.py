"""MuJoCo scene wrapper exposing privileged state for oracle planning."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import yaml

from .representations import Capability, ObjectState, ObjectType, OracleObservation


MODULE_DIR = Path(__file__).resolve().parent
ROOT_DIR = MODULE_DIR.parent
XML_PATH = MODULE_DIR / "blocked_passage.xml"
DEPLOY_CONFIG_PATH = ROOT_DIR / "deploy/deploy_mujoco/go2_arx5/config.yaml"


class BlockedPassageEnv:
    """Deterministic first scenario with a movable box blocking a corridor."""

    def __init__(self, capability: Capability | None = None) -> None:
        self.capability = capability or Capability()
        self.model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        self.data = mujoco.MjData(self.model)
        with DEPLOY_CONFIG_PATH.open(encoding="utf-8") as config_file:
            self.deploy_cfg = yaml.safe_load(config_file)

        self.robot_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "root"
        )
        if self.robot_joint_id < 0:
            self.robot_joint_id = 0
        self.box_joint_id = self._joint_id("movable_box_joint")
        self.box_body_id = self._body_id("movable_box")
        self.goal_body_id = self._body_id("goal_marker")
        self.robot_qpos_adr = int(self.model.jnt_qposadr[self.robot_joint_id])
        self.box_qpos_adr = int(self.model.jnt_qposadr[self.box_joint_id])
        self.rng = np.random.default_rng()

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

    def reset(self, seed: int | None = None) -> OracleObservation:
        self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        default_angles = np.asarray(
            self.deploy_cfg["default_angles"], dtype=np.float64
        )
        for index, joint_name in enumerate(
            (
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
        ):
            joint_id = self._joint_id(joint_name)
            qpos_adr = self.model.jnt_qposadr[joint_id]
            self.data.qpos[qpos_adr] = default_angles[index]

        start_y = float(self.rng.uniform(-0.12, 0.12))
        start_yaw = float(self.rng.uniform(-0.08, 0.08))
        self._set_free_joint(
            self.robot_qpos_adr,
            np.array([-2.30, start_y, 0.445]),
            start_yaw,
        )
        box_x = float(self.rng.uniform(-0.12, 0.12))
        box_y = float(self.rng.uniform(-0.02, 0.02))
        self._set_free_joint(
            self.box_qpos_adr, np.array([box_x, box_y, 0.30]), 0.0
        )
        self.data.qvel.fill(0.0)
        mujoco.mj_forward(self.model, self.data)
        return self.observe()

    def _set_free_joint(
        self, qpos_adr: int, position: np.ndarray, yaw: float
    ) -> None:
        self.data.qpos[qpos_adr : qpos_adr + 3] = position
        self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = [
            np.cos(yaw / 2.0),
            0.0,
            0.0,
            np.sin(yaw / 2.0),
        ]

    def set_robot_pose(self, x: float, y: float, yaw: float) -> None:
        """Set the base pose for kinematic planning visualization."""
        self._set_free_joint(
            self.robot_qpos_adr, np.array([x, y, 0.445]), yaw
        )
        robot_dof_adr = int(self.model.jnt_dofadr[self.robot_joint_id])
        self.data.qvel[robot_dof_adr : robot_dof_adr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def set_box_pose(self, x: float, y: float, yaw: float = 0.0) -> None:
        """Set an oracle counterfactual box pose without simulating a push."""
        self._set_free_joint(
            self.box_qpos_adr, np.array([x, y, 0.30]), yaw
        )
        box_dof_adr = int(self.model.jnt_dofadr[self.box_joint_id])
        self.data.qvel[box_dof_adr : box_dof_adr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    @property
    def push_target(self) -> np.ndarray:
        return np.array([2.45, 0.0, 0.0], dtype=np.float32)

    def observe(self) -> OracleObservation:
        robot_qpos = self.data.qpos[
            self.robot_qpos_adr : self.robot_qpos_adr + 7
        ]
        robot_yaw = 2.0 * np.arctan2(robot_qpos[6], robot_qpos[3])
        robot_state = np.zeros(12, dtype=np.float32)
        robot_state[:3] = robot_qpos[:3]
        robot_state[5] = robot_yaw
        robot_state[10] = 4.0
        robot_state[11] = 1.0

        box_qpos = self.data.qpos[self.box_qpos_adr : self.box_qpos_adr + 7]
        box_yaw = 2.0 * np.arctan2(box_qpos[6], box_qpos[3])
        objects = (
            ObjectState(
                1,
                ObjectType.STATIC_OBSTACLE,
                np.array([0.0, 0.9, 0.3]),
                np.array([6.0, 0.2, 0.6]),
            ),
            ObjectState(
                2,
                ObjectType.STATIC_OBSTACLE,
                np.array([0.0, -0.9, 0.3]),
                np.array([6.0, 0.2, 0.6]),
            ),
            ObjectState(
                3,
                ObjectType.STATIC_OBSTACLE,
                np.array([-3.0, 0.0, 0.3]),
                np.array([0.2, 2.0, 0.6]),
            ),
            ObjectState(
                4,
                ObjectType.STATIC_OBSTACLE,
                np.array([3.0, 0.0, 0.3]),
                np.array([0.2, 2.0, 0.6]),
            ),
            ObjectState(
                10,
                ObjectType.MOVABLE_BOX,
                box_qpos[:3].copy(),
                np.array([0.44, 1.44, 0.60]),
                yaw=float(box_yaw),
                movable=True,
                supportable=True,
                mass=5.0,
            ),
        )
        goal = np.array([1.80, 0.0, 0.0, 1.0], dtype=np.float32)
        return OracleObservation(
            robot_state=robot_state,
            goal=goal,
            objects=objects,
            capability=self.capability,
        )
