"""Common lifecycle and command types for executable high-level skills."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..representations import OracleObservation, SkillAction


class SkillStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class SkillCommand:
    velocity: np.ndarray
    end_effector_pose: np.ndarray


class Skill(ABC):
    status: SkillStatus = SkillStatus.IDLE

    @abstractmethod
    def can_execute(
        self, observation: OracleObservation, action: SkillAction
    ) -> bool:
        """Return whether the action is valid for this skill."""

    @abstractmethod
    def reset(self, action: SkillAction) -> None:
        """Start executing a new action."""

    @abstractmethod
    def step(self, observation: OracleObservation) -> SkillCommand:
        """Produce one low-level command and update the skill status."""
