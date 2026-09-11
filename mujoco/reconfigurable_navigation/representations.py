"""Typed interfaces shared by skills, planners, and future world models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


class SkillType(IntEnum):
    NAV = 0
    PUSH = 1
    CLIMB = 2
    JUMP = 3
    STOP = 4


class ObjectType(IntEnum):
    FLOOR = 0
    MOVABLE_BOX = 1
    STATIC_OBSTACLE = 2
    PLATFORM = 3
    BRIDGE = 4
    OTHER = 5


@dataclass(frozen=True)
class Capability:
    body_length: float = 0.48
    body_width: float = 0.30
    max_step_height: float = 0.28
    max_gap_width: float = 0.10
    max_slope: float = 0.25
    max_push_force: float = 80.0
    max_pushable_mass: float = 8.0
    max_linear_speed: float = 0.55

    def as_array(self) -> np.ndarray:
        return np.array(
            [
                self.body_length,
                self.body_width,
                self.max_step_height,
                self.max_gap_width,
                self.max_slope,
                self.max_push_force / 100.0,
                self.max_pushable_mass,
                self.max_linear_speed,
            ],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class ObjectState:
    object_id: int
    object_type: ObjectType
    center: np.ndarray
    size: np.ndarray
    yaw: float = 0.0
    movable: bool = False
    supportable: bool = False
    mass: float = 0.0
    confidence: float = 1.0
    linear_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    angular_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    climb_entry_pose: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )
    climb_landing_pose: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )


@dataclass(frozen=True)
class OracleObservation:
    robot_state: np.ndarray
    goal: np.ndarray
    objects: tuple[ObjectState, ...]
    capability: Capability
    state_valid: bool = True
    contact_object_ids: tuple[int, ...] = ()
    end_effector_contact_object_ids: tuple[int, ...] = ()
    body_contact_object_ids: tuple[int, ...] = ()
    illegal_collision: bool = False


@dataclass(frozen=True)
class SkillAction:
    skill: SkillType
    target_pose: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    object_id: int = -1
    support_id: int = -1

    def __post_init__(self) -> None:
        target_pose = np.asarray(self.target_pose, dtype=np.float32)
        if target_pose.shape not in ((0,), (3,), (4,)):
            raise ValueError("target_pose must have 0, 3, or 4 elements")
        object.__setattr__(self, "target_pose", target_pose)
