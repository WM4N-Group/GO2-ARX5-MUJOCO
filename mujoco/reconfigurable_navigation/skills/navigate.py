"""Target-pose navigation controller for the trained locomotion policy."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..representations import OracleObservation, SkillAction, SkillType
from .base import Skill, SkillCommand, SkillStatus


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class NavigateConfig:
    position_tolerance: float = 0.12
    yaw_tolerance: float = 0.15
    min_linear_speed: float = 0.15
    max_forward_speed: float = 0.45
    max_lateral_speed: float = 0.25
    max_yaw_rate: float = 0.70
    position_gain: float = 0.8
    yaw_gain: float = 1.2
    timeout_steps: int = 1500


class NavigateSkill(Skill):
    """Convert a world-frame target pose into body-frame velocity commands."""

    def __init__(self, config: NavigateConfig | None = None) -> None:
        self.config = config or NavigateConfig()
        self.status = SkillStatus.IDLE
        self.action: SkillAction | None = None
        self.steps = 0

    def can_execute(
        self, observation: OracleObservation, action: SkillAction
    ) -> bool:
        return bool(
            observation.state_valid
            and action.skill == SkillType.NAV
            and action.target_pose.shape == (3,)
            and np.isfinite(action.target_pose).all()
        )

    def reset(self, action: SkillAction) -> None:
        if action.skill != SkillType.NAV or action.target_pose.shape != (3,):
            raise ValueError("NavigateSkill requires a NAV action with [x, y, yaw]")
        self.action = action
        self.steps = 0
        self.status = SkillStatus.RUNNING
        self.failure_reason = None

    def step(self, observation: OracleObservation) -> SkillCommand:
        if self.status != SkillStatus.RUNNING or self.action is None:
            raise RuntimeError("NavigateSkill must be reset before step")
        if not observation.state_valid:
            self.status = SkillStatus.FAILED
            self.failure_reason = "invalid_robot_state"
            return self._stop_command()

        self.steps += 1
        robot_xy = observation.robot_state[:2]
        robot_yaw = float(observation.robot_state[5])
        error_world = self.action.target_pose[:2] - robot_xy
        distance = float(np.linalg.norm(error_world))
        final_yaw_error = wrap_angle(
            float(self.action.target_pose[2]) - robot_yaw
        )
        if distance <= self.config.position_tolerance:
            if abs(final_yaw_error) <= self.config.yaw_tolerance:
                self.status = SkillStatus.SUCCEEDED
                return self._stop_command()
            velocity = np.array(
                [
                    0.0,
                    0.0,
                    np.clip(
                        self.config.yaw_gain * final_yaw_error,
                        -self.config.max_yaw_rate,
                        self.config.max_yaw_rate,
                    ),
                ],
                dtype=np.float32,
            )
        else:
            cos_yaw = math.cos(robot_yaw)
            sin_yaw = math.sin(robot_yaw)
            error_body = np.array(
                [
                    cos_yaw * error_world[0] + sin_yaw * error_world[1],
                    -sin_yaw * error_world[0] + cos_yaw * error_world[1],
                ]
            )
            travel_heading = math.atan2(error_world[1], error_world[0])
            heading_error = wrap_angle(travel_heading - robot_yaw)
            planar_velocity = np.array(
                [
                    np.clip(
                        self.config.position_gain * error_body[0],
                        -self.config.max_forward_speed,
                        self.config.max_forward_speed,
                    ),
                    np.clip(
                        self.config.position_gain * error_body[1],
                        -self.config.max_lateral_speed,
                        self.config.max_lateral_speed,
                    ),
                ]
            )
            planar_speed = float(np.linalg.norm(planar_velocity))
            if 0.0 < planar_speed < self.config.min_linear_speed:
                planar_velocity *= self.config.min_linear_speed / planar_speed
            velocity = np.array(
                [
                    planar_velocity[0],
                    planar_velocity[1],
                    np.clip(
                        self.config.yaw_gain * heading_error,
                        -self.config.max_yaw_rate,
                        self.config.max_yaw_rate,
                    ),
                ],
                dtype=np.float32,
            )

        if self.steps >= self.config.timeout_steps:
            self.status = SkillStatus.FAILED
            self.failure_reason = "timeout"
            return self._stop_command()
        return SkillCommand(velocity, self._default_ee_pose())

    @staticmethod
    def _default_ee_pose() -> np.ndarray:
        return np.array(
            [0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0], dtype=np.float32
        )

    def _stop_command(self) -> SkillCommand:
        return SkillCommand(np.zeros(3, dtype=np.float32), self._default_ee_pose())
