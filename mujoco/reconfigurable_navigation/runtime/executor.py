"""Receding-horizon executor for structured navigation skills."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from ..climb_runtime import ClimbRuntime
from ..data import SkillTransition
from ..data.events import SkillEventRecorder
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
from .skill_backend import SkillExecutionBackend


StepCallback = Callable[
    [SkillAction | None, Skill | None, OracleObservation], bool
]
EventCallback = Callable[[str], None]
TransitionCallback = Callable[[SkillTransition], None]
SkillStartCallback = Callable[[SkillAction, SkillType | None], None]


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
        *,
        skill_backend: SkillExecutionBackend | None = None,
        stabilization_duration: float = 3.0,
        terminal_hold: bool = True,
    ) -> None:
        if climb_transition_duration < 0.0:
            raise ValueError("climb_transition_duration must be non-negative")
        if not np.isfinite(stabilization_duration) or stabilization_duration < 0.0:
            raise ValueError("stabilization_duration must be finite and non-negative")
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
        self.skill_backend = skill_backend
        self.stabilization_steps = round(stabilization_duration / runtime.control_dt)
        self.terminal_hold = terminal_hold
        self.climb_transition_steps = round(
            climb_transition_duration / runtime.control_dt
        )

    def run(
        self,
        on_step: StepCallback | None = None,
        on_event: EventCallback | None = None,
        on_transition: TransitionCallback | None = None,
        on_skill_start: SkillStartCallback | None = None,
    ) -> ExecutionResult:
        records: list[SkillExecutionRecord] = []
        replans = 0
        skill_failures = 0

        self._event(on_event, f"STABILIZE duration={self.stabilization_steps * self.runtime.control_dt:.2f}s")
        for _ in range(self.stabilization_steps):
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
                if decision.succeeded and self.terminal_hold:
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
            observation_before = deepcopy(observation) if on_transition is not None else None
            action_before = deepcopy(action) if on_transition is not None else None
            started_at = float(self.env.data.time) if on_transition is not None else 0.0
            previous_skill = records[-1].action.skill if records else None
            if on_skill_start is not None:
                on_skill_start(deepcopy(action), previous_skill)
            skill.reset(action)
            self._event(on_event, f"EXECUTE skill={action.skill.name}")
            event_recorder = SkillEventRecorder() if on_transition is not None else None
            status, steps, interrupted = self._execute_skill(
                action, skill, on_step, on_event, event_recorder
            )
            records.append(
                SkillExecutionRecord(action=action, status=status, steps=steps)
            )
            if on_transition is not None:
                assert observation_before is not None and action_before is not None
                assert event_recorder is not None
                on_transition(
                    SkillTransition(
                        action=action_before,
                        observation_before=observation_before,
                        observation_after=deepcopy(self.env.observe()),
                        status=status,
                        skill_steps=steps,
                        started_at=started_at,
                        ended_at=float(self.env.data.time),
                        control_dt=self.runtime.control_dt,
                        previous_skill=previous_skill,
                        interrupted=interrupted,
                        events=tuple(event_recorder.events),
                        event_sample_count=event_recorder.sample_count,
                        termination_reason="execution_interrupted" if interrupted else skill.failure_reason,
                    )
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
        event_recorder: SkillEventRecorder | None = None,
    ) -> tuple[SkillStatus, int, bool]:
        timeout_steps = skill.config.timeout_steps
        preparing = action.skill == SkillType.CLIMB and self.active_runtime != SkillType.CLIMB
        control_steps = 0
        if event_recorder is not None:
            event_recorder.sample(
                self.env.observe(), self.env.data.time, 0,
                self.skill_backend.initial_stage(action) if self.skill_backend is not None else "preparation" if preparing else "execution",
                skill.phase.value if isinstance(skill, PushSkill) else None,
            )

        def after_step(stage: str, phase: str | None = None) -> bool:
            nonlocal control_steps
            control_steps += 1
            if event_recorder is None:
                return self._step(on_step, action, skill)
            observation = self.env.observe()
            event_recorder.sample(
                observation, self.env.data.time, control_steps, stage,
                phase if phase is not None else skill.phase.value if isinstance(skill, PushSkill) else None,
            )
            return on_step is None or on_step(action, skill, observation)

        if self.skill_backend is not None:
            self.active_runtime = action.skill
            return self.skill_backend.execute_skill(action, skill, after_step)

        if preparing:
            self._event(
                on_event,
                "PREPARE skill=CLIMB "
                f"duration={self.climb_transition_steps * self.runtime.control_dt:.2f}s",
            )
            for _ in range(self.climb_transition_steps):
                self.runtime.hold_default()
                if not after_step("preparation"):
                    return SkillStatus.FAILED, 0, True
        self._activate_runtime(action.skill)
        previous_phase = skill.phase if isinstance(skill, PushSkill) else None
        if previous_phase is not None:
            self._event(on_event, f"PHASE skill=PUSH phase={previous_phase.value}")
        for step in range(timeout_steps + 1):
            command = skill.step(self.env.observe())
            self._advance_runtime(action.skill, command)
            if not after_step("execution"):
                return SkillStatus.FAILED, step + 1, True
            if isinstance(skill, PushSkill) and skill.phase != previous_phase:
                self._event(on_event, f"PHASE skill=PUSH phase={skill.phase.value}")
                previous_phase = skill.phase
            if skill.status != SkillStatus.RUNNING:
                return skill.status, step + 1, False
        skill.failure_reason = "executor_timeout"
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
        if self.skill_backend is not None:
            return self.skill_backend.create_skill(action)
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