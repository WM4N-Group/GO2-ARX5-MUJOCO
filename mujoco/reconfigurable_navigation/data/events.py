"""Contact and safety events sampled at control boundaries."""

from __future__ import annotations

from dataclasses import dataclass

from ..representations import OracleObservation


EVENT_KINDS = (
    "contact",
    "end_effector_contact",
    "body_contact",
    "illegal_collision",
    "invalid_robot_state",
)


@dataclass(frozen=True)
class SkillEvent:
    kind: str
    sim_time: float
    control_step: int
    stage: str
    active: bool
    object_id: int | None = None
    skill_phase: str | None = None


class SkillEventRecorder:
    def __init__(self) -> None:
        self.events: list[SkillEvent] = []
        self.sample_count = 0
        self._active: set[tuple[str, int | None]] = set()
        self._phase: str | None = None

    def sample(
        self,
        observation: OracleObservation,
        sim_time: float,
        control_step: int,
        stage: str,
        skill_phase: str | None = None,
    ) -> None:
        if skill_phase is not None and skill_phase != self._phase:
            self.events.append(SkillEvent(
                kind="phase", sim_time=float(sim_time), control_step=control_step,
                stage=stage, active=True, skill_phase=skill_phase,
            ))
        self._phase = skill_phase
        active: set[tuple[str, int | None]] = set()
        for kind, object_ids in (
            ("contact", observation.contact_object_ids),
            ("end_effector_contact", observation.end_effector_contact_object_ids),
            ("body_contact", observation.body_contact_object_ids),
        ):
            active.update((kind, object_id) for object_id in object_ids)
        if observation.illegal_collision:
            active.add(("illegal_collision", None))
        if not observation.state_valid:
            active.add(("invalid_robot_state", None))
        changes = sorted(
            self._active ^ active,
            key=lambda key: (key[0], -1 if key[1] is None else key[1]),
        )
        for kind, object_id in changes:
            self.events.append(
                SkillEvent(
                    kind=kind,
                    sim_time=float(sim_time),
                    control_step=control_step,
                    stage=stage,
                    active=(kind, object_id) in active,
                    object_id=object_id,
                    skill_phase=skill_phase,
                )
            )
        self._active = active
        self.sample_count += 1