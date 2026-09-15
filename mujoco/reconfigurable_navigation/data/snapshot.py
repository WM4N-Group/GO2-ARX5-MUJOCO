"""Independent in-memory snapshots at control or skill boundaries."""

from __future__ import annotations

from contextlib import contextmanager
from copy import copy, deepcopy
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import random
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch

from ..representations import SkillAction, SkillType
from ..skills.base import SkillStatus
from .events import SkillEventRecorder
from .transition import SkillTransition

if TYPE_CHECKING:
    from ..runtime.executor import ReconfigurableExecutor


def _rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().numpy().copy(),
    }


def _set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(torch.from_numpy(state["torch_cpu"]).clone())


@contextmanager
def _rng_scope(state: dict):
    previous = _rng_state()
    try:
        _set_rng_state(state)
        yield
    finally:
        _set_rng_state(previous)


def _clone_executor(source: ReconfigurableExecutor) -> ReconfigurableExecutor:
    model = source.env.model
    data = source.env.data
    runtimes = [source.runtime]
    if source.climb_runtime is not None:
        runtimes.append(source.climb_runtime)
    if source.skill_backend is not None:
        runtimes.extend(source.skill_backend.snapshot_runtimes())
    runtimes = list({id(runtime): runtime for runtime in runtimes}.values())
    if any(runtime.model is not model or runtime.data is not data for runtime in runtimes):
        raise ValueError("Snapshot runtimes must share the environment physics")
    if model.nplugin:
        raise ValueError("Plugin-backed models are not supported by this snapshot version")
    for callback_name in ("control", "passive", "sensor", "contactfilter", "act_dyn", "act_gain", "act_bias"):
        getter = getattr(mujoco, f"get_mjcb_{callback_name}", None)
        if getter is not None and getter() is not None:
            raise ValueError("Global MuJoCo callbacks are not supported by snapshots")

    cloned_model = copy(model)
    cloned_data = mujoco.MjData(cloned_model)
    mujoco.mj_copyData(cloned_data, cloned_model, data)
    memo = {id(model): cloned_model, id(data): cloned_data}
    for runtime in runtimes:
        if getattr(runtime, "on_step", None) is not None:
            raise ValueError("Snapshot capture requires a boundary without active runtime callbacks")
        if runtime.policy.training:
            raise ValueError("Snapshot policies must be in evaluation mode")
        tensors = (*runtime.policy.parameters(), *runtime.policy.buffers())
        if any(tensor.device.type != "cpu" for tensor in tensors):
            raise ValueError("Snapshot policies must use CPU tensors")
        buffer = BytesIO()
        torch.jit.save(runtime.policy, buffer)
        buffer.seek(0)
        policy = torch.jit.load(buffer, map_location="cpu")
        policy.eval()
        memo[id(runtime.policy)] = policy
    return deepcopy(source, memo)


@dataclass(frozen=True)
class SimulatorSnapshot:
    """Restore by forking; never rewind the caller's live physical world."""

    _executor: ReconfigurableExecutor
    _random_state: dict
    _previous_skill: SkillType | None = None

    @classmethod
    def capture(
        cls,
        executor: ReconfigurableExecutor,
        previous_skill: SkillType | None = None,
    ) -> SimulatorSnapshot:
        state = _rng_state()
        with _rng_scope(state):
            frozen = _clone_executor(executor)
        return cls(frozen, state, previous_skill)

    @property
    def time(self) -> float:
        return float(self._executor.env.data.time)

    def integration_state(self) -> np.ndarray:
        model = self._executor.env.model
        state_type = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(model, state_type), dtype=np.float64)
        mujoco.mj_getState(model, self._executor.env.data, state, state_type)
        return state

    def fork(self) -> ReconfigurableExecutor:
        with _rng_scope(self._random_state):
            return _clone_executor(self._executor)

    def rollout(
        self,
        action: SkillAction,
        max_control_steps: int | None = None,
    ) -> SkillReplay:
        if max_control_steps is not None and max_control_steps < 1:
            raise ValueError("max_control_steps must be positive")
        with _rng_scope(self._random_state):
            branch = self.fork()
            before = branch.env.observe()
            failure = branch.safety.observation_failure(before)
            if failure is not None:
                return SkillReplay(None, None, failure)
            action = deepcopy(action)
            skill = branch._create_skill(action)
            if skill is None:
                return SkillReplay(None, None, "unsupported_skill")
            if not skill.can_execute(before, action):
                return SkillReplay(None, None, "skill_not_executable")
            observation_before = deepcopy(before)
            action_before = deepcopy(action)
            started_at = float(branch.env.data.time)
            control_steps = 0

            def within_budget(_action, active_skill, _observation) -> bool:
                nonlocal control_steps
                control_steps += 1
                return (
                    max_control_steps is None
                    or control_steps < max_control_steps
                    or active_skill.status != SkillStatus.RUNNING
                )

            skill.reset(action)
            event_recorder = SkillEventRecorder()
            status, steps, truncated = branch._execute_skill(
                action, skill, within_budget, None, event_recorder
            )
            transition = SkillTransition(
                action=action_before,
                observation_before=observation_before,
                observation_after=deepcopy(branch.env.observe()),
                status=status,
                skill_steps=steps,
                started_at=started_at,
                ended_at=float(branch.env.data.time),
                control_dt=branch.runtime.control_dt,
                previous_skill=self._previous_skill,
                interrupted=truncated,
                events=tuple(event_recorder.events),
                event_sample_count=event_recorder.sample_count,
                termination_reason="rollout_budget_exhausted" if truncated else skill.failure_reason,
            )
            successor = None if truncated else SimulatorSnapshot.capture(branch, action.skill)
            reason = "rollout_budget_exhausted" if truncated else status.value
            return SkillReplay(transition, successor, reason)

    def save(self, path: Path | str) -> None:
        from .snapshot_io import save_snapshot

        save_snapshot(self, path)

    @classmethod
    def load(cls, path: Path | str, *, trusted: bool = False) -> SimulatorSnapshot:
        from .snapshot_io import load_snapshot

        with _rng_scope(_rng_state()):
            return load_snapshot(path, trusted=trusted)


@dataclass(frozen=True)
class SkillReplay:
    transition: SkillTransition | None
    next_snapshot: SimulatorSnapshot | None
    reason: str