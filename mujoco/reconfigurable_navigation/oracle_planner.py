"""Rule-based oracle planner used as the world-model data teacher."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .occupancy import GridConfig, OccupancyGrid
from .representations import ObjectState, OracleObservation, SkillAction, SkillType


@dataclass(frozen=True)
class PlanResult:
    actions: tuple[SkillAction, ...]
    direct_path: list[np.ndarray] | None
    reason: str


class OraclePlanner:
    def __init__(self, grid_config: GridConfig | None = None) -> None:
        self.grid = OccupancyGrid(grid_config)

    @staticmethod
    def robot_radius(observation: OracleObservation) -> float:
        capability = observation.capability
        return 0.5 * math.hypot(
            capability.body_length, capability.body_width
        )

    def path(
        self,
        observation: OracleObservation,
        excluded_ids: frozenset[int] = frozenset(),
    ) -> list[np.ndarray] | None:
        self.grid.rasterize(
            observation.objects,
            self.robot_radius(observation),
            excluded_ids=excluded_ids,
        )
        return self.grid.astar(
            observation.robot_state[:2], observation.goal[:2]
        )

    def blocking_objects(
        self, observation: OracleObservation
    ) -> tuple[ObjectState, ...]:
        if self.path(observation) is not None:
            return ()
        blockers = []
        for obj in observation.objects:
            if not obj.movable:
                continue
            if self.path(observation, frozenset({obj.object_id})) is not None:
                blockers.append(obj)
        return tuple(blockers)

    def plan(
        self,
        observation: OracleObservation,
        push_target: np.ndarray,
    ) -> PlanResult:
        direct_path = self.path(observation)
        goal_pose = np.array(
            [observation.goal[0], observation.goal[1], 0.0], dtype=np.float32
        )
        if direct_path is not None:
            return PlanResult(
                actions=(
                    SkillAction(SkillType.NAV, goal_pose),
                    SkillAction(SkillType.STOP),
                ),
                direct_path=direct_path,
                reason="goal_directly_reachable",
            )

        blockers = self.blocking_objects(observation)
        if not blockers:
            return PlanResult((), None, "no_single_movable_blocker")

        blocker = min(
            blockers,
            key=lambda obj: np.linalg.norm(
                obj.center[:2] - observation.robot_state[:2]
            ),
        )
        if blocker.mass > observation.capability.max_pushable_mass:
            return PlanResult((), None, "blocking_object_too_heavy")

        target = np.asarray(push_target, dtype=np.float32)
        push_delta = target[:2] - blocker.center[:2]
        push_distance = float(np.linalg.norm(push_delta))
        if push_distance < 1.0e-6:
            return PlanResult((), None, "push_target_matches_object_pose")
        push_direction = push_delta / push_distance
        approach_clearance = (
            blocker.size[0] / 2.0
            + observation.capability.body_length / 2.0
            + 0.12
        )
        approach_xy = blocker.center[:2] - push_direction * approach_clearance
        approach_yaw = math.atan2(push_direction[1], push_direction[0])
        approach_pose = np.array(
            [approach_xy[0], approach_xy[1], approach_yaw], dtype=np.float32
        )

        self.grid.rasterize(
            observation.objects, self.robot_radius(observation)
        )
        approach_path = self.grid.astar(
            observation.robot_state[:2], approach_xy
        )
        if approach_path is None:
            return PlanResult((), None, "blocker_has_no_reachable_push_face")

        return PlanResult(
            actions=(
                SkillAction(SkillType.NAV, approach_pose),
                SkillAction(
                    SkillType.PUSH,
                    target,
                    object_id=blocker.object_id,
                ),
                SkillAction(SkillType.NAV, goal_pose),
                SkillAction(SkillType.STOP),
            ),
            direct_path=None,
            reason="reconfiguration_required",
        )
