"""Closed-loop whole-body pushing for movable objects."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np

from ..representations import ObjectState, OracleObservation, SkillAction, SkillType
from .base import Skill, SkillCommand, SkillStatus
from .navigate import wrap_angle


class PushPhase(Enum):
    ALIGN = "align"
    CONTACT = "contact"
    PUSH = "push"
    VERIFY = "verify"
    RETREAT = "retreat"


@dataclass(frozen=True)
class PushConfig:
    align_steps: int = 100
    contact_timeout_steps: int = 750
    no_progress_timeout_steps: int = 750
    verify_steps: int = 50
    retreat_steps: int = 75
    timeout_steps: int = 4500
    push_speed: float = 0.30
    nominal_follow_speed: float = 0.20
    min_follow_speed: float = 0.15
    retreat_speed: float = 0.20
    desired_body_separation: float = 0.65
    min_body_separation: float = 0.60
    separation_gain: float = 2.0
    max_lateral_speed: float = 0.18
    max_yaw_rate: float = 0.40
    lateral_gain: float = 1.20
    yaw_gain: float = 1.50
    target_tolerance: float = 0.12
    lateral_tolerance: float = 0.18
    progress_epsilon: float = 0.005


class PushSkill(Skill):
    """Push one movable object to a target pose using arm contact and locomotion."""

    def __init__(self, config: PushConfig | None = None) -> None:
        self.config = config or PushConfig()
        self.status = SkillStatus.IDLE
        self.phase = PushPhase.ALIGN
        self.action: SkillAction | None = None
        self.steps = 0
        self.phase_steps = 0
        self.no_progress_steps = 0
        self.initial_object_xy: np.ndarray | None = None
        self.push_direction: np.ndarray | None = None
        self.push_right: np.ndarray | None = None
        self.target_progress = 0.0
        self.best_progress = 0.0

    def can_execute(
        self, observation: OracleObservation, action: SkillAction
    ) -> bool:
        obj = self._object(observation, action.object_id)
        return bool(
            observation.state_valid
            and not observation.illegal_collision
            and action.skill == SkillType.PUSH
            and action.target_pose.shape == (3,)
            and np.isfinite(action.target_pose).all()
            and obj is not None
            and obj.movable
            and obj.mass <= observation.capability.max_pushable_mass
        )

    def reset(self, action: SkillAction) -> None:
        if action.skill != SkillType.PUSH or action.target_pose.shape != (3,):
            raise ValueError("PushSkill requires PUSH with object target [x, y, yaw]")
        if action.object_id < 0:
            raise ValueError("PushSkill requires a valid object_id")
        self.action = action
        self.status = SkillStatus.RUNNING
        self.failure_reason = None
        self.phase = PushPhase.ALIGN
        self.steps = 0
        self.phase_steps = 0
        self.no_progress_steps = 0
        self.initial_object_xy = None
        self.push_direction = None
        self.push_right = None
        self.target_progress = 0.0
        self.best_progress = 0.0

    def step(self, observation: OracleObservation) -> SkillCommand:
        if self.status != SkillStatus.RUNNING or self.action is None:
            raise RuntimeError("PushSkill must be reset before step")
        if not observation.state_valid:
            return self._fail("invalid_robot_state")
        if observation.illegal_collision:
            return self._fail("illegal_collision")

        obj = self._object(observation, self.action.object_id)
        if obj is None:
            return self._fail("object_missing")
        if not obj.movable:
            return self._fail("object_not_movable")
        if obj.mass > observation.capability.max_pushable_mass:
            return self._fail("object_too_heavy")
        if self.action.object_id in observation.body_contact_object_ids:
            return self._fail("body_contact")

        if self.initial_object_xy is None:
            if not self._initialize_geometry(obj):
                return self._fail("degenerate_push_target")

        self.steps += 1
        self.phase_steps += 1
        if self.steps > self.config.timeout_steps:
            return self._fail("timeout")

        progress, lateral_error = self._progress(obj)
        if progress >= self.best_progress + self.config.progress_epsilon:
            self.best_progress = progress
            self.no_progress_steps = 0
        elif self.phase == PushPhase.PUSH:
            self.no_progress_steps += 1

        if self.phase == PushPhase.ALIGN:
            if self.phase_steps >= self.config.align_steps:
                self._set_phase(PushPhase.CONTACT)
            return self._align_command(observation)

        fingertip_contact = (
            self.action.object_id in observation.end_effector_contact_object_ids
        )
        if self.phase == PushPhase.CONTACT:
            if fingertip_contact:
                self._set_phase(PushPhase.PUSH)
            elif self.phase_steps > self.config.contact_timeout_steps:
                return self._fail("contact_timeout")
            return self._push_command(observation, obj)

        if self.phase == PushPhase.PUSH:
            if self._target_reached(progress, lateral_error):
                self._set_phase(PushPhase.VERIFY)
                return self._stop_command(push_pose=True)
            if self.no_progress_steps > self.config.no_progress_timeout_steps:
                return self._fail("no_progress")
            return self._push_command(observation, obj)

        if self.phase == PushPhase.VERIFY:
            if self.phase_steps >= self.config.verify_steps:
                if not self._target_reached(progress, lateral_error):
                    return self._fail("target_not_maintained")
                self._set_phase(PushPhase.RETREAT)
            return self._stop_command(push_pose=True)

        if self.phase_steps >= self.config.retreat_steps:
            self.status = SkillStatus.SUCCEEDED
            return self._stop_command(push_pose=False)
        return self._retreat_command(observation)

    def _initialize_geometry(self, obj: ObjectState) -> bool:
        target_xy = self.action.target_pose[:2]
        delta = target_xy - obj.center[:2]
        distance = float(np.linalg.norm(delta))
        if distance < 1.0e-6:
            return False
        self.initial_object_xy = obj.center[:2].copy()
        self.push_direction = delta / distance
        self.push_right = np.array(
            [-self.push_direction[1], self.push_direction[0]], dtype=np.float32
        )
        self.target_progress = distance
        return True

    def _progress(self, obj: ObjectState) -> tuple[float, float]:
        displacement = obj.center[:2] - self.initial_object_xy
        return (
            float(displacement @ self.push_direction),
            float((self.action.target_pose[:2] - obj.center[:2]) @ self.push_right),
        )

    def _target_reached(self, progress: float, lateral_error: float) -> bool:
        return bool(
            progress >= self.target_progress - self.config.target_tolerance
            and abs(lateral_error) <= self.config.lateral_tolerance
        )

    def _align_command(self, observation: OracleObservation) -> SkillCommand:
        heading = math.atan2(self.push_direction[1], self.push_direction[0])
        yaw_error = wrap_angle(heading - float(observation.robot_state[5]))
        return SkillCommand(
            np.array(
                [
                    0.0,
                    0.0,
                    np.clip(
                        self.config.yaw_gain * yaw_error,
                        -self.config.max_yaw_rate,
                        self.config.max_yaw_rate,
                    ),
                ],
                dtype=np.float32,
            ),
            self._push_pose(),
        )

    def _push_command(
        self, observation: OracleObservation, obj: ObjectState
    ) -> SkillCommand:
        relative_object = obj.center[:2] - observation.robot_state[:2]
        lateral_error = float(relative_object @ self.push_right)
        separation = float(relative_object @ self.push_direction)
        forward_speed = float(
            np.clip(
                self.config.nominal_follow_speed
                + self.config.separation_gain
                * (separation - self.config.desired_body_separation),
                0.0,
                self.config.push_speed,
            )
        )
        if (
            0.0 < forward_speed < self.config.min_follow_speed
            and separation > self.config.min_body_separation
        ):
            forward_speed = self.config.min_follow_speed
        if separation <= self.config.min_body_separation:
            forward_speed = 0.0
        lateral_speed = float(
            np.clip(
                self.config.lateral_gain * lateral_error,
                -self.config.max_lateral_speed,
                self.config.max_lateral_speed,
            )
        )
        velocity_world = (
            forward_speed * self.push_direction
            + lateral_speed * self.push_right
        )
        return SkillCommand(
            self._world_velocity_to_body(observation, velocity_world),
            self._push_pose(),
        )

    def _retreat_command(self, observation: OracleObservation) -> SkillCommand:
        velocity_world = -self.config.retreat_speed * self.push_direction
        return SkillCommand(
            self._world_velocity_to_body(observation, velocity_world),
            self._neutral_pose(),
        )

    def _world_velocity_to_body(
        self, observation: OracleObservation, velocity_world: np.ndarray
    ) -> np.ndarray:
        yaw = float(observation.robot_state[5])
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        heading = math.atan2(self.push_direction[1], self.push_direction[0])
        yaw_error = wrap_angle(heading - yaw)
        return np.array(
            [
                cos_yaw * velocity_world[0] + sin_yaw * velocity_world[1],
                -sin_yaw * velocity_world[0] + cos_yaw * velocity_world[1],
                np.clip(
                    self.config.yaw_gain * yaw_error,
                    -self.config.max_yaw_rate,
                    self.config.max_yaw_rate,
                ),
            ],
            dtype=np.float32,
        )

    def _set_phase(self, phase: PushPhase) -> None:
        self.phase = phase
        self.phase_steps = 0

    def _fail(self, reason: str) -> SkillCommand:
        self.status = SkillStatus.FAILED
        self.failure_reason = reason
        return self._stop_command(push_pose=False)

    @staticmethod
    def _object(
        observation: OracleObservation, object_id: int
    ) -> ObjectState | None:
        return next(
            (obj for obj in observation.objects if obj.object_id == object_id),
            None,
        )

    @staticmethod
    def _push_pose() -> np.ndarray:
        return np.array(
            [0.8, 0.0, 0.15, 1.0, 0.0, 0.0, 0.0], dtype=np.float32
        )

    @staticmethod
    def _neutral_pose() -> np.ndarray:
        return np.array(
            [0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0], dtype=np.float32
        )

    def _stop_command(self, push_pose: bool) -> SkillCommand:
        return SkillCommand(
            np.zeros(3, dtype=np.float32),
            self._push_pose() if push_pose else self._neutral_pose(),
        )