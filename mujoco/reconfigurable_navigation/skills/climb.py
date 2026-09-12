"""Forward CLIMB skill for reaching a higher support surface."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..representations import OracleObservation, SkillAction, SkillType
from .base import Skill, SkillCommand, SkillStatus
from .navigate import wrap_angle


@dataclass(frozen=True)
class ClimbConfig:
    forward_speed: float = 0.5
    position_tolerance: float = 0.15
    lateral_tolerance: float = 0.60
    height_tolerance: float = 0.05
    entry_yaw_tolerance: float = 0.25
    stable_steps: int = 5
    timeout_steps: int = 750


class ClimbSkill(Skill):
    """Drive the dedicated CLIMB actor until a raised target is reached."""

    def __init__(self, config: ClimbConfig | None = None) -> None:
        self.config = config or ClimbConfig()
        self.status = SkillStatus.IDLE
        self.action: SkillAction | None = None
        self.steps = 0
        self.stable_steps = 0
        self.start_xy: np.ndarray | None = None
        self.direction: np.ndarray | None = None
        self.target_progress = 0.0

    def can_execute(
        self, observation: OracleObservation, action: SkillAction
    ) -> bool:
        if (
            not observation.state_valid
            or observation.illegal_collision
            or action.skill != SkillType.CLIMB
            or action.target_pose.shape != (4,)
            or not np.isfinite(action.target_pose).all()
        ):
            return False
        delta = action.target_pose[:2] - observation.robot_state[:2]
        if np.linalg.norm(delta) <= self.config.position_tolerance:
            climb_heading = float(action.target_pose[3])
        else:
            climb_heading = float(np.arctan2(delta[1], delta[0]))
        yaw_error = wrap_angle(climb_heading - float(observation.robot_state[5]))
        return abs(yaw_error) <= self.config.entry_yaw_tolerance

    def reset(self, action: SkillAction) -> None:
        if action.skill != SkillType.CLIMB or action.target_pose.shape != (4,):
            raise ValueError("ClimbSkill requires CLIMB with [x, y, z, yaw]")
        self.action = action
        self.status = SkillStatus.RUNNING
        self.failure_reason = None
        self.steps = 0
        self.stable_steps = 0
        self.start_xy = None
        self.direction = None
        self.target_progress = 0.0

    def step(self, observation: OracleObservation) -> SkillCommand:
        if self.status != SkillStatus.RUNNING or self.action is None:
            raise RuntimeError("ClimbSkill must be reset before step")
        if not observation.state_valid:
            return self._fail("invalid_robot_state")
        if observation.illegal_collision:
            return self._fail("illegal_collision")

        if self.start_xy is None:
            self._initialize_geometry(observation)

        self.steps += 1
        displacement = observation.robot_state[:2] - self.start_xy
        progress = float(displacement @ self.direction)
        lateral_error = float(
            (self.action.target_pose[:2] - observation.robot_state[:2])
            @ np.array([-self.direction[1], self.direction[0]])
        )
        reached_position = bool(
            progress >= self.target_progress - self.config.position_tolerance
            and self._landing_is_safe(observation, lateral_error)
        )
        reached_height = bool(
            observation.robot_state[2]
            >= self.action.target_pose[2] - self.config.height_tolerance
        )
        if reached_position and reached_height:
            self.stable_steps += 1
            if self.stable_steps >= self.config.stable_steps:
                self.status = SkillStatus.SUCCEEDED
                return self._stop_command()
        else:
            self.stable_steps = 0

        if self.steps >= self.config.timeout_steps:
            return self._fail("timeout")
        return SkillCommand(
            np.array([self.config.forward_speed, 0.0, 0.0], dtype=np.float32),
            self._neutral_pose(),
        )

    def _landing_is_safe(
        self,
        observation: OracleObservation,
        lateral_error: float,
    ) -> bool:
        support = next(
            (
                obj
                for obj in observation.objects
                if obj.object_id == self.action.support_id
                and obj.supportable
            ),
            None,
        )
        if support is None:
            return abs(lateral_error) <= self.config.lateral_tolerance
        delta = observation.robot_state[:2] - support.center[:2]
        cos_yaw = np.cos(support.yaw)
        sin_yaw = np.sin(support.yaw)
        local = np.array(
            [
                cos_yaw * delta[0] + sin_yaw * delta[1],
                -sin_yaw * delta[0] + cos_yaw * delta[1],
            ]
        )
        robot_radius = 0.5 * np.hypot(
            observation.capability.body_length,
            observation.capability.body_width,
        )
        safe_half_size = support.size[:2] / 2.0 - robot_radius
        return bool(
            np.all(safe_half_size > 0.0)
            and np.all(np.abs(local) <= safe_half_size)
        )

    def _initialize_geometry(self, observation: OracleObservation) -> None:
        self.start_xy = observation.robot_state[:2].copy()
        delta = self.action.target_pose[:2] - self.start_xy
        self.target_progress = float(np.linalg.norm(delta))
        if self.target_progress <= self.config.position_tolerance:
            yaw = float(self.action.target_pose[3])
            self.direction = np.array([np.cos(yaw), np.sin(yaw)])
        else:
            self.direction = delta / self.target_progress

    def _fail(self, reason: str) -> SkillCommand:
        self.status = SkillStatus.FAILED
        self.failure_reason = reason
        return self._stop_command()

    @staticmethod
    def _neutral_pose() -> np.ndarray:
        return np.array(
            [0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0], dtype=np.float32
        )

    def _stop_command(self) -> SkillCommand:
        return SkillCommand(np.zeros(3, dtype=np.float32), self._neutral_pose())