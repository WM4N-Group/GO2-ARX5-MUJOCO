"""Safety checks and execution budgets for closed-loop skill execution."""

from __future__ import annotations

from dataclasses import dataclass

from ..representations import OracleObservation


@dataclass(frozen=True)
class SafetyConfig:
    max_replans: int = 12
    max_skill_executions: int = 8
    max_skill_failures: int = 2


class SafetyMonitor:
    def __init__(self, config: SafetyConfig | None = None) -> None:
        self.config = config or SafetyConfig()

    @staticmethod
    def observation_failure(observation: OracleObservation) -> str | None:
        if not observation.state_valid:
            return "invalid_robot_state"
        if observation.illegal_collision:
            return "illegal_collision"
        return None

    def budget_failure(
        self,
        replans: int,
        skill_executions: int,
        skill_failures: int,
    ) -> str | None:
        if replans >= self.config.max_replans:
            return "replan_budget_exhausted"
        if skill_executions >= self.config.max_skill_executions:
            return "skill_budget_exhausted"
        if skill_failures >= self.config.max_skill_failures:
            return "skill_failure_budget_exhausted"
        return None