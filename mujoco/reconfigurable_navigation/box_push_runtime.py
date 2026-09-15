"""MuJoCo counterpart of the hybrid PUSH arm controller."""

from __future__ import annotations

from collections import deque
from pathlib import Path
import runpy

import mujoco
import numpy as np
import torch

from .box_climb_runtime import BoxClimbRuntime
from .locomotion_runtime import MJ_POLICY_INDICES, POLICY_MJ_INDICES, projected_gravity


CONTROL_PATH = Path(__file__).resolve().parents[2] / "source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/mdp/box_push_control.py"
CONTROL = runpy.run_path(str(CONTROL_PATH))


def batch(value):
    return torch.as_tensor(np.asarray(value), dtype=torch.float64).unsqueeze(0)


class BoxPushArmController:
    def __init__(self, runtime, box_geom: int, goal: np.ndarray):
        self.runtime = runtime
        self.model, self.data = runtime.model, runtime.data
        self.box_geom = box_geom
        self.box_body = int(self.model.geom_bodyid[box_geom])
        self.box_size = self.model.geom_size[box_geom].copy() * 2.0
        self.goal = np.asarray(goal, dtype=np.float64).copy()
        self.hand_body = runtime._body_id("x5_link6")
        self.mount_body = runtime._body_id("x5_base_link")
        self.hand_bodies = {runtime._body_id(f"x5_link{index}") for index in (6, 7, 8)}
        self.arm_qpos = runtime.joint_qpos_adr[12:]
        self.arm_dof = runtime.joint_dof_adr[12:]
        joint_ids = self.model.dof_jntid[self.arm_dof]
        midpoint = self.model.jnt_range[joint_ids].mean(axis=1)
        half_range = 0.45 * np.ptp(self.model.jnt_range[joint_ids], axis=1)
        self.lower, self.upper = midpoint - half_range, midpoint + half_range
        self.gravity_data = mujoco.MjData(self.model)
        self.reset()

    def reset(self):
        self.reference = self.data.qpos[self.arm_qpos].copy()
        self.locked_reference = self.reference.copy()
        self.locked = False
        self.motion_ready = False
        self.approached = False
        self.pose_override = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    def hand_position(self):
        rotation = self.data.xmat[self.hand_body].reshape(3, 3)
        return self.data.xpos[self.hand_body] + rotation @ np.array([0.15, 0.0, 0.0])

    def box_rotation(self):
        return self.data.geom_xmat[self.box_geom].reshape(3, 3)

    def at_goal(self):
        return self.goal[0] - self.data.geom_xpos[self.box_geom, 0] < 0.10

    def hand_target(self):
        position = self.data.geom_xpos[self.box_geom]
        rotation = self.box_rotation()
        local = rotation.T @ (self.hand_position() - position)
        self.approached |= bool(CONTROL["side_approach_ready"](batch(local), self.box_size[0] / 2.0, 0.10)[0])
        self.approached &= abs(local[2]) < 0.06
        press = 0.02 if self.approached and not self.at_goal() else -0.10
        return position + rotation @ np.array([-self.box_size[0] / 2.0 + press, 0.0, 0.0])

    def contacts(self):
        hand_force = np.zeros(3)
        hand_total_force = np.zeros(3)
        positions = []
        other_forces = {}
        wrench = np.zeros(6)
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if self.box_geom not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == self.box_geom else contact.geom1
            body = int(self.model.geom_bodyid[other])
            if body == 0:
                continue
            mujoco.mj_contactForce(self.model, self.data, index, wrench)
            force = contact.frame.reshape(3, 3)[0] * wrench[0]
            total_force = contact.frame.reshape(3, 3).T @ wrench[:3]
            if other == contact.geom1:
                force = -force
                total_force = -total_force
            if body in self.hand_bodies:
                hand_force += force
                hand_total_force += total_force
                if np.linalg.norm(force) > 0.0:
                    positions.append(contact.pos.copy())
            else:
                other_forces[body] = other_forces.get(body, np.zeros(3)) + force
        position = np.mean(positions, axis=0) if positions else np.full(3, np.nan)
        local = self.box_rotation().T @ (position - self.data.geom_xpos[self.box_geom])
        near_tip = np.linalg.norm(position - self.hand_position()) < 0.07
        valid = bool(CONTROL["valid_front_contact"](
            batch(hand_force), batch(self.box_rotation()[:, 0]), batch(local),
            torch.tensor([bool(near_tip)], dtype=torch.bool), self.box_size,
        )[0])
        return {
            "valid": valid,
            "invalid_hand": bool(np.linalg.norm(hand_force) > 1.0 and not valid),
            "forbidden": any(np.linalg.norm(force) > 1.0 for force in other_forces.values()),
            "force": hand_force,
            "total_force": hand_total_force,
            "position": position,
        }

    def apply(self, leg_actions, *, target=None):
        if np.shape(leg_actions) != (12,):
            raise ValueError("Hybrid PUSH requires 12 leg actions")
        position = self.hand_position()
        target = self.hand_target() if target is None else np.asarray(target, dtype=np.float64)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("Arm target requires three finite world coordinates")
        if self.contacts()["valid"] and not self.locked and not self.at_goal():
            self.locked_reference = self.reference.copy()
            self.pose_override[:3] = self.data.xmat[self.mount_body].reshape(3, 3).T @ (position - self.data.xpos[self.mount_body])
            self.locked = True
        self.locked &= not self.at_goal()
        self.motion_ready |= self.locked
        jacobian = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, jacobian, None, position, self.hand_body)
        arm_jacobian = jacobian[:, self.arm_dof]
        delta = arm_jacobian.T @ np.linalg.solve(arm_jacobian @ arm_jacobian.T + 0.01 * np.eye(3), target - position)
        desired = np.clip(self.data.qpos[self.arm_qpos] + delta, self.lower, self.upper)
        self.reference = CONTROL["coordinated_joint_target"](batch(self.reference), batch(desired), self.runtime.control_dt)[0].numpy()
        if self.locked:
            self.reference = self.locked_reference.copy()
        self.gravity_data.qpos[:] = self.data.qpos
        self.gravity_data.qvel.fill(0.0)
        mujoco.mj_forward(self.model, self.gravity_data)
        compensated = self.reference + self.gravity_data.qfrc_bias[self.arm_dof] / self.runtime.kp[12:]
        combined = np.zeros(18)
        if self.motion_ready:
            combined[:12] = leg_actions
        combined[12:] = (compensated - self.runtime.default_qpos[12:]) / self.runtime.action_scale[12:]
        return combined


class BoxPushRuntime(BoxClimbRuntime):
    def __init__(self, policy_path, *, model, data, box_geom, goal):
        super().__init__(policy_path, model=model, data=data)
        self.joint_velocity_limits[12:] = 3.0
        self.arm = BoxPushArmController(self, box_geom, goal)
        self.history = deque(maxlen=3)
        self.activate()

    def activate(self):
        super().activate()
        if hasattr(self, "arm"):
            self.arm.reset()
            frame = self._frame()
            self.history.clear()
            self.history.extend(frame.copy() for _ in range(3))

    def _yaw_rotation(self):
        rotation = self.data.xmat[self.base_id].reshape(3, 3)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        return np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])

    def _box_velocity(self):
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.arm.box_body, velocity, 0)
        return velocity[3:]

    def _commands(self):
        rotation = self._yaw_rotation()
        position = self.data.geom_xpos[self.arm.box_geom]
        face = position - self.arm.box_rotation()[:, 0] * self.arm.box_size[0] / 2.0
        relative_face = rotation.T @ (face - self.data.xpos[self.base_id])
        direction = rotation.T @ (self.arm.goal - position)
        contact = self.arm.contacts()
        velocity = CONTROL["contact_push_velocity"](
            batch(relative_face), batch(rotation.T @ self._box_velocity()),
            torch.tensor([self.arm.goal[0] - position[0]], dtype=torch.float64),
            torch.tensor([contact["valid"] and not contact["forbidden"]], dtype=torch.bool),
            torch.tensor([np.arctan2(direction[1], direction[0])], dtype=torch.float64),
        )[0].numpy()
        target = self.arm.hand_target()
        local = self.data.xmat[self.arm.mount_body].reshape(3, 3).T @ (target - self.data.xpos[self.arm.mount_body])
        pose = self.arm.pose_override.copy() if self.arm.locked else np.concatenate((local, [1.0, 0.0, 0.0, 0.0]))
        return velocity, pose

    def _frame(self):
        velocity, pose = self._commands()
        return np.concatenate((
            self.data.qvel[3:6] * 0.2, projected_gravity(self.data.xquat[self.base_id]),
            (self.data.qpos[self.joint_qpos_adr] - self.default_qpos)[MJ_POLICY_INDICES],
            self.data.qvel[self.joint_dof_adr][MJ_POLICY_INDICES] * 0.05,
            self.last_action, velocity, pose,
        )).astype(np.float32)

    def observation(self):
        slices = ((0, 3), (3, 6), (6, 24), (24, 42), (42, 60), (60, 63), (63, 70))
        history = np.concatenate([frame[start:end] for start, end in slices for frame in self.history])
        rotation = self._yaw_rotation()
        position = self.data.geom_xpos[self.arm.box_geom]
        box = np.concatenate((
            rotation.T @ (position - self.data.xpos[self.base_id]),
            rotation.T @ self._box_velocity(), rotation.T @ (self.arm.goal - position),
            (rotation.T @ self.arm.box_rotation()[:, 0])[:2], self.arm.box_size,
            [float(self.arm.locked), float(self.arm.contacts()["valid"])],
        ))
        observation = np.concatenate((history, box)).astype(np.float32)
        if observation.shape != (226,) or not np.isfinite(observation).all():
            raise ValueError("Hybrid PUSH requires 226 finite observations")
        return observation

    def step(self):
        with torch.inference_mode():
            actions = self.policy(torch.from_numpy(self.observation()).unsqueeze(0))[0].numpy()
        combined = self.arm.apply(actions)
        self._simulate_target(self.default_qpos + combined[POLICY_MJ_INDICES] * self.action_scale)
        self.last_action = combined.copy()
        self.history.append(self._frame())
        return combined.copy()