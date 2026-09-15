"""Runtime-specific execution of the shared structured skill interfaces."""

from collections.abc import Callable
from typing import Protocol

from ..representations import SkillAction
from ..skills.base import Skill, SkillStatus


ControlStepCallback = Callable[[str, str | None], bool]


class SkillExecutionBackend(Protocol):
    def snapshot_runtimes(self) -> tuple[object, ...]:
        ...

    def initial_stage(self, action: SkillAction) -> str:
        ...

    def create_skill(self, action: SkillAction) -> Skill | None:
        ...

    def execute_skill(
        self, action: SkillAction, skill: Skill, after_step: ControlStepCallback,
    ) -> tuple[SkillStatus, int, bool]:
        ...