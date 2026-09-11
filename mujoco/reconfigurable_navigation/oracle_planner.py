"""Rule-based oracle planner used as the world-model data teacher."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .occupancy import GridConfig, OccupancyGrid
from .representations import (
    ObjectState,
    ObjectType,
    OracleObservation,
    SkillAction,
    SkillType,
)


@dataclass(frozen=True)
class PlanResult:
    actions: tuple[SkillAction, ...]
    direct_path: list[np.ndarray] | None
    reason: str


@dataclass(frozen=True)
class PlannerConfig:
    goal_tolerance: float = 0.12
    approach_position_tolerance: float = 0.12
    approach_yaw_tolerance: float = 0.15
    nominal_base_clearance: float = 0.25
    support_position_margin: float = 0.15
    support_height_tolerance: float = 0.10


class OraclePlanner:
    def __init__(
        self,
        grid_config: GridConfig | None = None,
        config: PlannerConfig | None = None,
    ) -> None:
        self.grid = OccupancyGrid(grid_config)
        self.config = config or PlannerConfig()

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

    @staticmethod
    def _platform_top(platform: ObjectState) -> float:
        return float(platform.center[2] + platform.size[2] / 2.0)

    def _current_support_platform(
        self, observation: OracleObservation
    ) -> ObjectState | None:
        robot_xy = observation.robot_state[:2]
        robot_height = float(observation.robot_state[2])
        for platform in observation.objects:
            if (
                platform.object_type != ObjectType.PLATFORM
                or not platform.supportable
            ):
                continue
            delta = robot_xy - platform.center[:2]
            cos_yaw = math.cos(platform.yaw)
            sin_yaw = math.sin(platform.yaw)
            local = np.array(
                [
                    cos_yaw * delta[0] + sin_yaw * delta[1],
                    -sin_yaw * delta[0] + cos_yaw * delta[1],
                ]
            )
            half_size = platform.size[:2] / 2.0
            expected_height = (
                self._platform_top(platform)
                + self.config.nominal_base_clearance
            )
            if (
                np.all(np.abs(local) <= half_size + self.config.support_position_margin)
                and abs(robot_height - expected_height)
                <= self.config.support_height_tolerance
            ):
                return platform
        return None

    def _path_between(
        self,
        observation: OracleObservation,
        start_xy: np.ndarray,
        goal_xy: np.ndarray,
        excluded_ids: frozenset[int] = frozenset(),
    ) -> list[np.ndarray] | None:
        self.grid.rasterize(
            observation.objects,
            self.robot_radius(observation),
            excluded_ids=excluded_ids,
        )
        return self.grid.astar(start_xy, goal_xy)

    def _climb_plan(
        self,
        observation: OracleObservation,
        goal_pose: np.ndarray,
        push_target: np.ndarray,
    ) -> PlanResult | None:
        candidates: list[tuple[float, PlanResult]] = []
        platform_too_high = False
        for platform in observation.objects:
            if (
                platform.object_type != ObjectType.PLATFORM
                or not platform.supportable
                or platform.climb_entry_pose.shape != (3,)
                or platform.climb_landing_pose.shape != (3,)
                or not np.isfinite(platform.climb_entry_pose).all()
                or not np.isfinite(platform.climb_landing_pose).all()
            ):
                continue
            top_height = self._platform_top(platform)
            if top_height > observation.capability.max_step_height:
                platform_too_high = True
                continue
            landing_path = self._path_between(
                observation,
                platform.climb_landing_pose[:2],
                observation.goal[:2],
                frozenset({platform.object_id}),
            )
            if landing_path is None:
                continue
            approach_path = self._path_between(
                observation,
                observation.robot_state[:2],
                platform.climb_entry_pose[:2],
            )
            if approach_path is None:
                blockers = tuple(
                    obj
                    for obj in observation.objects
                    if obj.movable
                    and self._path_between(
                        observation,
                        observation.robot_state[:2],
                        platform.climb_entry_pose[:2],
                        frozenset({obj.object_id}),
                    )
                    is not None
                )
                if blockers:
                    blocker = min(
                        blockers,
                        key=lambda obj: np.linalg.norm(
                            obj.center[:2] - observation.robot_state[:2]
                        ),
                    )
                    climb_actions = (
                        SkillAction(
                            SkillType.NAV,
                            platform.climb_entry_pose,
                        ),
                        SkillAction(
                            SkillType.CLIMB,
                            np.array(
                                [
                                    platform.climb_landing_pose[0],
                                    platform.climb_landing_pose[1],
                                    top_height
                                    + self.config.nominal_base_clearance,
                                    platform.climb_landing_pose[2],
                                ],
                                dtype=np.float32,
                            ),
                            support_id=platform.object_id,
                        ),
                        SkillAction(SkillType.NAV, goal_pose),
                        SkillAction(SkillType.STOP),
                    )
                    return self._push_plan(
                        observation,
                        blocker,
                        push_target,
                        climb_actions,
                        "reconfiguration_for_climb",
                    )
                continue

            entry_position_error = float(
                np.linalg.norm(
                    platform.climb_entry_pose[:2]
                    - observation.robot_state[:2]
                )
            )
            entry_yaw_error = (
                float(platform.climb_entry_pose[2])
                - float(observation.robot_state[5])
                + math.pi
            ) % (2.0 * math.pi) - math.pi
            landing_pose = np.array(
                [
                    platform.climb_landing_pose[0],
                    platform.climb_landing_pose[1],
                    top_height + self.config.nominal_base_clearance,
                    platform.climb_landing_pose[2],
                ],
                dtype=np.float32,
            )
            actions = [
                SkillAction(
                    SkillType.CLIMB,
                    landing_pose,
                    support_id=platform.object_id,
                ),
                SkillAction(SkillType.NAV, goal_pose),
                SkillAction(SkillType.STOP),
            ]
            if (
                entry_position_error > self.config.approach_position_tolerance
                or abs(entry_yaw_error) > self.config.approach_yaw_tolerance
            ):
                actions.insert(
                    0,
                    SkillAction(
                        SkillType.NAV,
                        platform.climb_entry_pose,
                    ),
                )
            candidates.append(
                (
                    entry_position_error,
                    PlanResult(
                        actions=tuple(actions),
                        direct_path=approach_path,
                        reason="climb_required",
                    ),
                )
            )
        if candidates:
            return min(candidates, key=lambda candidate: candidate[0])[1]
        if platform_too_high:
            return PlanResult((), None, "platform_too_high")
        return None

    def _push_plan(
        self,
        observation: OracleObservation,
        blocker: ObjectState,
        push_target: np.ndarray,
        tail_actions: tuple[SkillAction, ...],
        reason: str,
    ) -> PlanResult:
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

        approach_path = self._path_between(
            observation,
            observation.robot_state[:2],
            approach_xy,
        )
        if approach_path is None:
            return PlanResult((), None, "blocker_has_no_reachable_push_face")

        approach_position_error = float(
            np.linalg.norm(approach_xy - observation.robot_state[:2])
        )
        approach_yaw_error = (
            approach_yaw - float(observation.robot_state[5]) + math.pi
        ) % (2.0 * math.pi) - math.pi
        actions = [
            SkillAction(
                SkillType.PUSH,
                target,
                object_id=blocker.object_id,
            ),
            *tail_actions,
        ]
        if (
            approach_position_error > self.config.approach_position_tolerance
            or abs(approach_yaw_error) > self.config.approach_yaw_tolerance
        ):
            actions.insert(0, SkillAction(SkillType.NAV, approach_pose))
        return PlanResult(
            actions=tuple(actions),
            direct_path=None,
            reason=reason,
        )

    def plan(
        self,
        observation: OracleObservation,
        push_target: np.ndarray,
    ) -> PlanResult:
        direct_path = self.path(observation)
        goal_pose = np.array(
            [observation.goal[0], observation.goal[1], 0.0], dtype=np.float32
        )
        if (
            np.linalg.norm(observation.goal[:2] - observation.robot_state[:2])
            <= self.config.goal_tolerance
        ):
            return PlanResult(
                actions=(SkillAction(SkillType.STOP),),
                direct_path=direct_path,
                reason="goal_reached",
            )
        support_platform = self._current_support_platform(observation)
        if support_platform is not None:
            support_path = self.path(
                observation, frozenset({support_platform.object_id})
            )
            if support_path is not None:
                return PlanResult(
                    actions=(
                        SkillAction(SkillType.NAV, goal_pose),
                        SkillAction(SkillType.STOP),
                    ),
                    direct_path=support_path,
                    reason="goal_reachable_from_platform",
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

        climb_plan = self._climb_plan(observation, goal_pose, push_target)
        if climb_plan is not None:
            return climb_plan

        blockers = self.blocking_objects(observation)
        if not blockers:
            return PlanResult((), None, "no_single_movable_blocker")

        blocker = min(
            blockers,
            key=lambda obj: np.linalg.norm(
                obj.center[:2] - observation.robot_state[:2]
            ),
        )
        return self._push_plan(
            observation,
            blocker,
            push_target,
            (
            SkillAction(SkillType.NAV, goal_pose),
            SkillAction(SkillType.STOP),
            ),
            "reconfiguration_required",
        )
