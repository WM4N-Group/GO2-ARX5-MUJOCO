"""Run deterministic checks for the first reconfigurable-navigation slice."""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np

from reconfigurable_navigation.env import BlockedPassageEnv
from reconfigurable_navigation.oracle_planner import OraclePlanner
from reconfigurable_navigation.representations import Capability, SkillType
from reconfigurable_navigation.skills import NavigateSkill, SkillStatus


def check_seed(seed: int) -> None:
    env = BlockedPassageEnv()
    observation = env.reset(seed=seed)
    planner = OraclePlanner()

    assert planner.path(observation) is None
    assert [obj.object_id for obj in planner.blocking_objects(observation)] == [10]

    plan = planner.plan(observation, env.push_target)
    assert plan.reason == "reconfiguration_required"
    assert [action.skill for action in plan.actions] == [
        SkillType.NAV,
        SkillType.PUSH,
        SkillType.NAV,
        SkillType.STOP,
    ]
    assert plan.actions[1].object_id == 10

    env.set_box_pose(*env.push_target)
    post_push_observation = env.observe()
    assert planner.path(post_push_observation) is not None
    post_push_plan = planner.plan(post_push_observation, env.push_target)
    assert post_push_plan.reason == "goal_directly_reachable"


def check_capability_limit() -> None:
    env = BlockedPassageEnv(
        capability=Capability(max_pushable_mass=4.0)
    )
    observation = env.reset(seed=0)
    plan = OraclePlanner().plan(observation, env.push_target)
    assert plan.actions == ()
    assert plan.reason == "blocking_object_too_heavy"


def check_navigate_skill() -> None:
    env = BlockedPassageEnv()
    observation = env.reset(seed=0)
    nav_action = OraclePlanner().plan(observation, env.push_target).actions[0]
    skill = NavigateSkill()
    assert skill.can_execute(observation, nav_action)
    skill.reset(nav_action)

    command = skill.step(observation)
    assert skill.status == SkillStatus.RUNNING
    assert command.velocity.shape == (3,)
    assert abs(command.velocity[0]) <= skill.config.max_forward_speed
    assert abs(command.velocity[1]) <= skill.config.max_lateral_speed
    assert abs(command.velocity[2]) <= skill.config.max_yaw_rate

    reached_state = observation.robot_state.copy()
    reached_state[0] = nav_action.target_pose[0]
    reached_state[1] = nav_action.target_pose[1]
    reached_state[5] = nav_action.target_pose[2]
    reached_observation = replace(observation, robot_state=reached_state)
    stop_command = skill.step(reached_observation)
    assert skill.status == SkillStatus.SUCCEEDED
    assert np.array_equal(stop_command.velocity, np.zeros(3))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=100)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")

    for seed in range(args.seeds):
        check_seed(seed)
    check_capability_limit()
    check_navigate_skill()
    print(
        f"Oracle checks passed for {args.seeds} randomized scenes; "
        "reachability, capability limits, and NAV commands verified."
    )


if __name__ == "__main__":
    main()
