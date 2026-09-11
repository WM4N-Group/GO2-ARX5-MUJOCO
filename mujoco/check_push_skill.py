"""Evaluate the physical NAV-to-PUSH pipeline across randomized scenes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from reconfigurable_navigation.env import BlockedPassageEnv
from reconfigurable_navigation.locomotion_runtime import LocomotionRuntime
from reconfigurable_navigation.oracle_planner import OraclePlanner
from reconfigurable_navigation.representations import (
    OracleObservation,
    SkillAction,
    SkillType,
)
from reconfigurable_navigation.runtime import ReconfigurableExecutor
from reconfigurable_navigation.skills import Skill


MAX_MANIPULATOR_PENETRATION = 0.01


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    execution_succeeded: bool
    reason: str
    skill_sequence: tuple[SkillType, ...]
    replans: int
    fingertip_contact: bool
    box_progress: float
    post_push_reachable: bool
    goal_error: float
    illegal_collision: bool
    body_contact: bool
    max_manipulator_penetration: float
    min_base_height: float

    @property
    def succeeded(self) -> bool:
        return bool(
            self.execution_succeeded
            and self.fingertip_contact
            and self.post_push_reachable
            and not self.illegal_collision
            and not self.body_contact
            and self.max_manipulator_penetration
            <= MAX_MANIPULATOR_PENETRATION
        )


def run_episode(seed: int) -> EpisodeResult:
    env = BlockedPassageEnv()
    env.reset(seed=seed)
    runtime = LocomotionRuntime(env)
    planner = OraclePlanner()
    executor = ReconfigurableExecutor(env, runtime)
    box_start = next(
        obj for obj in env.observe().objects if obj.object_id == 10
    ).center.copy()
    min_base_height = float(env.observe().robot_state[2])
    illegal_collision = False
    fingertip_contact = False
    body_contact = False
    max_manipulator_penetration = 0.0

    def on_step(
        action: SkillAction | None,
        _skill: Skill | None,
        observation: OracleObservation,
    ) -> bool:
        nonlocal fingertip_contact
        nonlocal body_contact
        nonlocal illegal_collision
        nonlocal max_manipulator_penetration
        nonlocal min_base_height
        min_base_height = min(
            min_base_height, float(observation.robot_state[2])
        )
        illegal_collision = illegal_collision or observation.illegal_collision
        if action is not None and action.skill == SkillType.PUSH:
            fingertip_contact = fingertip_contact or (
                action.object_id in observation.end_effector_contact_object_ids
            )
            body_contact = body_contact or (
                action.object_id in observation.body_contact_object_ids
            )
            for contact in env.data.contact[: env.data.ncon]:
                if env.box_geom_id not in (int(contact.geom1), int(contact.geom2)):
                    continue
                other = (
                    int(contact.geom2)
                    if int(contact.geom1) == env.box_geom_id
                    else int(contact.geom1)
                )
                if other in env.manipulator_geom_ids:
                    max_manipulator_penetration = max(
                        max_manipulator_penetration,
                        -float(contact.dist),
                    )
        return True

    execution = executor.run(on_step=on_step)

    final_observation = env.observe()
    box = next(
        obj for obj in final_observation.objects if obj.object_id == 10
    )
    box_progress = float(box.center[0] - box_start[0])
    post_push_reachable = planner.path(final_observation) is not None
    goal_error = float(
        np.linalg.norm(
            final_observation.goal[:2] - final_observation.robot_state[:2]
        )
    )
    return EpisodeResult(
        seed=seed,
        execution_succeeded=execution.succeeded,
        reason=execution.reason,
        skill_sequence=tuple(
            record.action.skill for record in execution.records
        ),
        replans=execution.replans,
        fingertip_contact=fingertip_contact,
        box_progress=box_progress,
        post_push_reachable=post_push_reachable,
        goal_error=goal_error,
        illegal_collision=illegal_collision,
        body_contact=body_contact,
        max_manipulator_penetration=max_manipulator_penetration,
        min_base_height=min_base_height,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--min-success-rate", type=float, default=0.85)
    args = parser.parse_args()
    if args.seeds <= 0:
        parser.error("--seeds must be positive")
    if not 0.0 <= args.min_success_rate <= 1.0:
        parser.error("--min-success-rate must be between 0 and 1")

    results = [run_episode(seed) for seed in range(args.seeds)]
    for result in results:
        print(
            f"seed={result.seed} success={result.succeeded} "
            f"reason={result.reason} "
            f"skills={','.join(skill.name for skill in result.skill_sequence)} "
            f"replans={result.replans} "
            f"fingertip_contact={result.fingertip_contact} "
            f"box_progress={result.box_progress:.3f} "
            f"reachable={result.post_push_reachable} "
            f"goal_error={result.goal_error:.3f} "
            f"illegal_collision={result.illegal_collision} "
            f"body_contact={result.body_contact} "
            f"max_penetration={result.max_manipulator_penetration:.4f} "
            f"min_base_height={result.min_base_height:.3f}"
        )

    successes = sum(result.succeeded for result in results)
    success_rate = successes / len(results)
    print(
        f"PUSH evaluation: {successes}/{len(results)} succeeded "
        f"({success_rate:.1%}); required {args.min_success_rate:.1%}"
    )
    raise SystemExit(0 if success_rate >= args.min_success_rate else 1)


if __name__ == "__main__":
    main()