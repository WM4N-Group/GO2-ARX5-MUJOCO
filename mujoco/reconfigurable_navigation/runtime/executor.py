"""Receding-horizon executor for structured navigation skills."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..climb_runtime import ClimbRuntime
from ..env import BlockedPassageEnv
from ..locomotion_runtime import LocomotionRuntime
from ..representations import OracleObservation, SkillAction, SkillType
from ..skills import (
    ClimbSkill,
    NavigateSkill,
    PushSkill,
    Skill,
    SkillCommand,
    SkillStatus,
)
from .replanner import OracleReplanner
from .safety import SafetyMonitor


StepCallback = Callable[
    [SkillAction | None, Skill | None, OracleObservation], bool
]
EventCallback = Callable[[str], None]


@dataclass(frozen=True)
class SkillExecutionRecord:
    action: SkillAction
    status: SkillStatus
    steps: int


@dataclass(frozen=True)
class ExecutionResult:
    succeeded: bool
    reason: str
    records: tuple[SkillExecutionRecord, ...]
    replans: int
    skill_failures: int


class ReconfigurableExecutor:
    def __init__(
        self,
        env: BlockedPassageEnv,
        runtime: LocomotionRuntime,
        climb_runtime: ClimbRuntime | None = None,
        replanner: OracleReplanner | None = None,
        safety: SafetyMonitor | None = None,
        climb_transition_duration: float = 0.5,
    ) -> None:
        if climb_transition_duration < 0.0:
            raise ValueError("climb_transition_duration must be non-negative")
        self.env = env
        self.runtime = runtime
        self.climb_runtime = climb_runtime
        if climb_runtime is not None:
            if (
                climb_runtime.model is not runtime.model
                or climb_runtime.data is not runtime.data
            ):
                raise ValueError("CLIMB and locomotion runtimes must share physics")
            if climb_runtime.control_dt != runtime.control_dt:
                raise ValueError("CLIMB and locomotion control periods must match")
        self.replanner = replanner or OracleReplanner(env.push_target)
        self.safety = safety or SafetyMonitor()
        self.active_runtime = SkillType.NAV
        self.climb_transition_steps = round(
            climb_transition_duration / runtime.control_dt
        )

    def run(
        self,
        on_step: StepCallback | None = None,
        on_event: EventCallback | None = None,
    ) -> ExecutionResult:
        records: list[SkillExecutionRecord] = []
        replans = 0
        skill_failures = 0

        self._event(on_event, "STABILIZE duration=3.00s")
        for _ in range(round(3.0 / self.runtime.control_dt)):
            self.runtime.hold_default()
            if not self._step(on_step, None, None):
                return self._result(
                    False, "execution_interrupted", records, replans, skill_failures
                )

        while True:
            observation = self.env.observe()
            failure = self.safety.observation_failure(observation)
            if failure is not None:
                return self._result(
                    False, failure, records, replans, skill_failures
                )
            failure = self.safety.budget_failure(
                replans, len(records), skill_failures
            )
            if failure is not None:
                return self._result(
                    False, failure, records, replans, skill_failures
                )

            decision = self.replanner.decide(observation)
            replans += 1
            skills = ",".join(
                action.skill.name for action in decision.plan.actions
            )
            self._event(
                on_event,
                f"REPLAN index={replans} reason={decision.reason} plan={skills or 'NONE'}",
            )
            if decision.terminal:
                if decision.succeeded:
                    self._activate_runtime(SkillType.NAV)
                    self.runtime.step(
                        np.zeros(3, dtype=np.float32),
                        np.array(
                            [0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
                            dtype=np.float32,
                        ),
                    )
                    if not self._step(on_step, decision.action, None):
                        return self._result(
                            False,
                            "execution_interrupted",
                            records,
                            replans,
                            skill_failures,
                        )
                return self._result(
                    decision.succeeded,
                    decision.reason,
                    records,
                    replans,
                    skill_failures,
                )

            action = decision.action
            if action is None:
                return self._result(
                    False,
                    "replanner_returned_no_action",
                    records,
                    replans,
                    skill_failures,
                )
            skill = self._create_skill(action)
            if skill is None or not skill.can_execute(observation, action):
                return self._result(
                    False,
                    "skill_not_executable",
                    records,
                    replans,
                    skill_failures,
                )
            skill.reset(action)
            self._event(on_event, f"EXECUTE skill={action.skill.name}")
            status, steps, interrupted = self._execute_skill(
                action, skill, on_step, on_event
            )
            records.append(
                SkillExecutionRecord(action=action, status=status, steps=steps)
            )
            self._event(
                on_event,
                f"FINISH skill={action.skill.name} status={status.value} "
                f"duration={steps * self.runtime.control_dt:.2f}s",
            )
            if interrupted:
                return self._result(
                    False,
                    "execution_interrupted",
                    records,
                    replans,
                    skill_failures,
                )
            if status == SkillStatus.FAILED:
                skill_failures += 1

    def _execute_skill(
        self,
        action: SkillAction,
        skill: Skill,
        on_step: StepCallback | None,
        on_event: EventCallback | None,
    ) -> tuple[SkillStatus, int, bool]:
        timeout_steps = skill.config.timeout_steps
        if action.skill == SkillType.CLIMB and self.active_runtime != SkillType.CLIMB:
            self._event(
                on_event,
                "PREPARE skill=CLIMB "
                f"duration={self.climb_transition_steps * self.runtime.control_dt:.2f}s",
            )
            for _ in range(self.climb_transition_steps):
                self.runtime.hold_default()
                if not self._step(on_step, action, skill):
                    return SkillStatus.FAILED, 0, True
        self._activate_runtime(action.skill)
        previous_phase = skill.phase if isinstance(skill, PushSkill) else None
        if previous_phase is not None:
            self._event(on_event, f"PHASE skill=PUSH phase={previous_phase.value}")
        for step in range(timeout_steps + 1):
            command = skill.step(self.env.observe())
            self._advance_runtime(action.skill, command)
            if not self._step(on_step, action, skill):
                return SkillStatus.FAILED, step + 1, True
            if isinstance(skill, PushSkill) and skill.phase != previous_phase:
                self._event(on_event, f"PHASE skill=PUSH phase={skill.phase.value}")
                previous_phase = skill.phase
            if skill.status != SkillStatus.RUNNING:
                return skill.status, step + 1, False
        return SkillStatus.FAILED, timeout_steps + 1, False

    def _step(
        self,
        callback: StepCallback | None,
        action: SkillAction | None,
        skill: Skill | None,
    ) -> bool:
        return callback is None or callback(action, skill, self.env.observe())

    def _activate_runtime(self, skill_type: SkillType) -> None:
        target_runtime = (
            SkillType.CLIMB if skill_type == SkillType.CLIMB else SkillType.NAV
        )
        if target_runtime == self.active_runtime:
            return
        if target_runtime == SkillType.CLIMB:
            if self.climb_runtime is None:
                raise RuntimeError("CLIMB runtime is not configured")
            self.climb_runtime.activate()
        else:
            self.runtime.reset()
        self.active_runtime = target_runtime

    def _advance_runtime(
        self, skill_type: SkillType, command: SkillCommand
    ) -> None:
        if skill_type == SkillType.CLIMB:
            if self.climb_runtime is None:
                raise RuntimeError("CLIMB runtime is not configured")
            self.climb_runtime.step(command.velocity)
        else:
            self.runtime.step(command.velocity, command.end_effector_pose)

    def _create_skill(self, action: SkillAction | None) -> Skill | None:
        if action is None:
            return None
        if action.skill == SkillType.NAV:
            return NavigateSkill()
        if action.skill == SkillType.PUSH:
            return PushSkill()
        if action.skill == SkillType.CLIMB and self.climb_runtime is not None:
            return ClimbSkill()
        return None

    @staticmethod
    def _event(callback: EventCallback | None, message: str) -> None:
        if callback is not None:
            callback(message)

    @staticmethod
    def _result(
        succeeded: bool,
        reason: str,
        records: list[SkillExecutionRecord],
        replans: int,
        skill_failures: int,
    ) -> ExecutionResult:
        return ExecutionResult(
            succeeded=succeeded,
            reason=reason,
            records=tuple(records),
            replans=replans,
            skill_failures=skill_failures,
        )