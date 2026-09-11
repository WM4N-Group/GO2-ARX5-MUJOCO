"""Convert each fresh Oracle plan into one receding-horizon decision."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..oracle_planner import OraclePlanner, PlanResult
from ..representations import OracleObservation, SkillAction, SkillType


@dataclass(frozen=True)
class ReplanDecision:
    plan: PlanResult
    action: SkillAction | None
    terminal: bool
    succeeded: bool
    reason: str


class OracleReplanner:
    def __init__(
        self,
        push_target: np.ndarray,
        planner: OraclePlanner | None = None,
    ) -> None:
        self.push_target = np.asarray(push_target, dtype=np.float32)
        self.planner = planner or OraclePlanner()

    def decide(self, observation: OracleObservation) -> ReplanDecision:
        plan = self.planner.plan(observation, self.push_target)
        if not plan.actions:
            return ReplanDecision(
                plan=plan,
                action=None,
                terminal=True,
                succeeded=False,
                reason=plan.reason,
            )
        action = plan.actions[0]
        if action.skill == SkillType.STOP:
            return ReplanDecision(
                plan=plan,
                action=action,
                terminal=True,
                succeeded=plan.reason == "goal_reached",
                reason=plan.reason,
            )
        return ReplanDecision(
            plan=plan,
            action=action,
            terminal=False,
            succeeded=False,
            reason=plan.reason,
        )