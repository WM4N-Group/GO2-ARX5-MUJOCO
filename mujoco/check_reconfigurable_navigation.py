"""Run deterministic checks for the first reconfigurable-navigation slice."""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np

from reconfigurable_navigation.env import BlockedPassageEnv
from reconfigurable_navigation.oracle_planner import OraclePlanner
from reconfigurable_navigation.representations import (
    Capability,
    ObjectState,
    ObjectType,
    SkillType,
)
from reconfigurable_navigation.runtime import (
    OracleReplanner,
    SafetyConfig,
    SafetyMonitor,
)
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

    reached_approach_state = observation.robot_state.copy()
    reached_approach_state[:2] = plan.actions[0].target_pose[:2]
    reached_approach_state[5] = plan.actions[0].target_pose[2]
    reached_approach_observation = replace(
        observation, robot_state=reached_approach_state
    )
    approach_plan = planner.plan(
        reached_approach_observation, env.push_target
    )
    assert approach_plan.actions[0].skill == SkillType.PUSH

    env.set_box_pose(*env.push_target)
    post_push_observation = env.observe()
    assert planner.path(post_push_observation) is not None
    post_push_plan = planner.plan(post_push_observation, env.push_target)
    assert post_push_plan.reason == "goal_directly_reachable"

    reached_goal_state = post_push_observation.robot_state.copy()
    reached_goal_state[:2] = post_push_observation.goal[:2]
    reached_goal_observation = replace(
        post_push_observation, robot_state=reached_goal_state
    )
    goal_plan = planner.plan(reached_goal_observation, env.push_target)
    assert goal_plan.actions[0].skill == SkillType.STOP
    assert goal_plan.reason == "goal_reached"


def check_capability_limit() -> None:
    env = BlockedPassageEnv(
        capability=Capability(max_pushable_mass=4.0)
    )
    observation = env.reset(seed=0)
    plan = OraclePlanner().plan(observation, env.push_target)
    assert plan.actions == ()
    assert plan.reason == "blocking_object_too_heavy"


def check_platform_plan() -> None:
    observation = BlockedPassageEnv().reset(seed=0)
    platform = ObjectState(
        object_id=20,
        object_type=ObjectType.PLATFORM,
        center=np.array([0.8, 0.0, 0.08], dtype=np.float32),
        size=np.array([3.2, 1.4, 0.16], dtype=np.float32),
        supportable=True,
        climb_entry_pose=np.array([-1.5, 0.0, 0.0], dtype=np.float32),
        climb_landing_pose=np.array([1.3, 0.0, 0.0], dtype=np.float32),
    )
    robot_state = observation.robot_state.copy()
    robot_state[:3] = [-2.3, 0.0, 0.28]
    robot_state[5] = 0.0
    goal = np.array([2.0, 0.0, 0.0, 1.0], dtype=np.float32)
    observation = replace(
        observation,
        robot_state=robot_state,
        goal=goal,
        objects=(platform,),
    )
    planner = OraclePlanner()
    plan = planner.plan(observation, np.zeros(3, dtype=np.float32))
    assert plan.reason == "climb_required"
    assert [action.skill for action in plan.actions] == [
        SkillType.NAV,
        SkillType.CLIMB,
        SkillType.NAV,
        SkillType.STOP,
    ]
    assert plan.actions[1].support_id == platform.object_id
    assert plan.actions[1].target_pose.shape == (4,)

    approach_state = robot_state.copy()
    approach_state[:2] = platform.climb_entry_pose[:2]
    approach_plan = planner.plan(
        replace(observation, robot_state=approach_state),
        np.zeros(3, dtype=np.float32),
    )
    assert approach_plan.actions[0].skill == SkillType.CLIMB

    landing_state = robot_state.copy()
    landing_state[:2] = platform.climb_landing_pose[:2]
    landing_state[2] = plan.actions[1].target_pose[2]
    landing_plan = planner.plan(
        replace(observation, robot_state=landing_state),
        np.zeros(3, dtype=np.float32),
    )
    assert landing_plan.reason == "goal_reachable_from_platform"
    assert landing_plan.actions[0].skill == SkillType.NAV

    goal_state = landing_state.copy()
    goal_state[:2] = goal[:2]
    goal_plan = planner.plan(
        replace(observation, robot_state=goal_state),
        np.zeros(3, dtype=np.float32),
    )
    assert goal_plan.reason == "goal_reached"
    assert goal_plan.actions[0].skill == SkillType.STOP

    low_capability_plan = planner.plan(
        replace(
            observation,
            capability=Capability(max_step_height=0.10),
        ),
        np.zeros(3, dtype=np.float32),
    )
    assert low_capability_plan.actions == ()
    assert low_capability_plan.reason == "platform_too_high"


def check_push_then_climb_plan() -> None:
    observation = BlockedPassageEnv().reset(seed=0)
    platform = ObjectState(
        object_id=20,
        object_type=ObjectType.PLATFORM,
        center=np.array([1.5, 0.0, 0.08], dtype=np.float32),
        size=np.array([2.4, 1.4, 0.16], dtype=np.float32),
        supportable=True,
        climb_entry_pose=np.array([-0.4, 0.0, 0.0], dtype=np.float32),
        climb_landing_pose=np.array([1.3, 0.0, 0.0], dtype=np.float32),
    )
    blocker = ObjectState(
        object_id=10,
        object_type=ObjectType.MOVABLE_BOX,
        center=np.array([-1.0, 0.0, 0.3], dtype=np.float32),
        size=np.array([0.44, 1.44, 0.60], dtype=np.float32),
        movable=True,
        supportable=True,
        mass=5.0,
    )
    walls = (
        ObjectState(
            1,
            ObjectType.STATIC_OBSTACLE,
            np.array([0.0, 0.9, 0.3]),
            np.array([6.0, 0.2, 0.6]),
        ),
        ObjectState(
            2,
            ObjectType.STATIC_OBSTACLE,
            np.array([0.0, -0.9, 0.3]),
            np.array([6.0, 0.2, 0.6]),
        ),
    )
    robot_state = observation.robot_state.copy()
    robot_state[:3] = [-2.3, 0.0, 0.28]
    robot_state[5] = 0.0
    observation = replace(
        observation,
        robot_state=robot_state,
        goal=np.array([2.0, 0.0, 0.0, 1.0], dtype=np.float32),
        objects=(*walls, blocker, platform),
    )
    push_target = np.array([2.8, 0.0, 0.0], dtype=np.float32)
    plan = OraclePlanner().plan(observation, push_target)
    assert plan.reason == "reconfiguration_for_climb"
    assert [action.skill for action in plan.actions] == [
        SkillType.NAV,
        SkillType.PUSH,
        SkillType.NAV,
        SkillType.CLIMB,
        SkillType.NAV,
        SkillType.STOP,
    ]
    assert plan.actions[1].object_id == blocker.object_id
    assert plan.actions[3].support_id == platform.object_id

    moved_blocker = replace(blocker, center=push_target.copy())
    post_push_observation = replace(
        observation,
        objects=(*walls, moved_blocker, platform),
    )
    post_push_plan = OraclePlanner().plan(
        post_push_observation, push_target
    )
    assert post_push_plan.reason == "climb_required"
    assert [action.skill for action in post_push_plan.actions] == [
        SkillType.NAV,
        SkillType.CLIMB,
        SkillType.NAV,
        SkillType.STOP,
    ]


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


def check_replanner_terminals() -> None:
    env = BlockedPassageEnv()
    observation = env.reset(seed=0)
    replanner = OracleReplanner(env.push_target)
    initial_decision = replanner.decide(observation)
    assert not initial_decision.terminal
    assert initial_decision.action.skill == SkillType.NAV

    env.set_box_pose(*env.push_target)
    reached_goal_state = env.observe().robot_state.copy()
    reached_goal_state[:2] = env.observe().goal[:2]
    reached_goal_observation = replace(
        env.observe(), robot_state=reached_goal_state
    )
    goal_decision = replanner.decide(reached_goal_observation)
    assert goal_decision.terminal
    assert goal_decision.succeeded
    assert goal_decision.action.skill == SkillType.STOP

    heavy_env = BlockedPassageEnv(
        capability=Capability(max_pushable_mass=4.0)
    )
    heavy_decision = OracleReplanner(heavy_env.push_target).decide(
        heavy_env.reset(seed=0)
    )
    assert heavy_decision.terminal
    assert not heavy_decision.succeeded
    assert heavy_decision.action is None
    assert heavy_decision.reason == "blocking_object_too_heavy"


def check_safety_monitor() -> None:
    observation = BlockedPassageEnv().reset(seed=0)
    monitor = SafetyMonitor(
        SafetyConfig(
            max_replans=2,
            max_skill_executions=3,
            max_skill_failures=1,
        )
    )
    assert monitor.observation_failure(observation) is None
    assert (
        monitor.observation_failure(replace(observation, state_valid=False))
        == "invalid_robot_state"
    )
    assert (
        monitor.observation_failure(
            replace(observation, illegal_collision=True)
        )
        == "illegal_collision"
    )
    assert monitor.budget_failure(2, 0, 0) == "replan_budget_exhausted"
    assert monitor.budget_failure(0, 3, 0) == "skill_budget_exhausted"
    assert (
        monitor.budget_failure(0, 0, 1)
        == "skill_failure_budget_exhausted"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=100)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")

    for seed in range(args.seeds):
        check_seed(seed)
    check_capability_limit()
    check_platform_plan()
    check_push_then_climb_plan()
    check_navigate_skill()
    check_replanner_terminals()
    check_safety_monitor()
    print(
        f"Oracle checks passed for {args.seeds} randomized scenes; "
        "reachability, capability limits, and NAV commands verified."
    )


if __name__ == "__main__":
    main()
