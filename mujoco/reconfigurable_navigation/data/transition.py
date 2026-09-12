"""Serializable observations at executed skill boundaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
import math

import numpy as np

from ..representations import OracleObservation, SkillAction, SkillType
from ..skills.base import SkillStatus
from .events import EVENT_KINDS, SkillEvent


def _json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"Unsupported transition value: {type(value).__name__}")


@dataclass(frozen=True)
class SkillTransition:
    action: SkillAction
    observation_before: OracleObservation
    observation_after: OracleObservation
    status: SkillStatus
    skill_steps: int
    started_at: float
    ended_at: float
    control_dt: float
    previous_skill: SkillType | None = None
    interrupted: bool = False
    events: tuple[SkillEvent, ...] = ()
    event_sample_count: int = 0
    termination_reason: str | None = None

    def __post_init__(self) -> None:
        if self.status not in (SkillStatus.SUCCEEDED, SkillStatus.FAILED):
            raise ValueError("Transition status must describe a finished skill")
        if self.skill_steps < 0:
            raise ValueError("skill_steps must be non-negative")
        if self.event_sample_count < 0 or (self.events and not self.event_sample_count):
            raise ValueError("Events require a positive sample count")
        if not all(math.isfinite(value) for value in (self.started_at, self.ended_at, self.control_dt)):
            raise ValueError("Transition times must be finite")
        if self.ended_at < self.started_at or self.control_dt <= 0.0:
            raise ValueError("Invalid transition time interval")

    @property
    def elapsed_sim_time(self) -> float:
        return self.ended_at - self.started_at

    def to_json(self, metadata: dict | None = None) -> str:
        payload = asdict(self)
        reason = self.termination_reason or (
            "execution_interrupted" if self.interrupted
            else "target_reached" if self.status == SkillStatus.SUCCEEDED
            else "unknown_failure"
        )
        payload.update(
            schema_version=2,
            snapshot_kind="skill_boundary_observation",
            outcome="interrupted" if self.interrupted else self.status.value,
            elapsed_sim_time=self.elapsed_sim_time,
            skill_success=None if self.interrupted else self.status == SkillStatus.SUCCEEDED,
            termination_reason=reason,
            failure_reason=reason if (
                not self.interrupted and self.status == SkillStatus.FAILED
                and reason != "unknown_failure"
            ) else None,
            event_sampling="control_boundary" if self.event_sample_count else None,
            process_label_scope=("observed_prefix" if self.interrupted else "executed_skill")
            if self.event_sample_count else None,
            process_labels={
                kind: any(event.kind == kind and event.active for event in self.events)
                if self.event_sample_count else None
                for kind in EVENT_KINDS
            },
            metadata={} if metadata is None else metadata,
        )
        return json.dumps(payload, default=_json_default, allow_nan=False)