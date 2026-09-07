"""Visualize the current oracle reconfiguration pipeline in MuJoCo."""

from __future__ import annotations

import argparse
import math
import time

import mujoco
import mujoco.viewer
import numpy as np

from reconfigurable_navigation.env import BlockedPassageEnv
from reconfigurable_navigation.oracle_planner import OraclePlanner


def pause(viewer: mujoco.viewer.Handle, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while viewer.is_running() and time.monotonic() < deadline:
        viewer.sync()
        time.sleep(1.0 / 60.0)
    return viewer.is_running()


def animate_robot_path(
    env: BlockedPassageEnv,
    viewer: mujoco.viewer.Handle,
    path: list[np.ndarray],
    speed: float,
) -> bool:
    if not path:
        return viewer.is_running()
    previous = env.observe().robot_state[:2].copy()
    yaw = float(env.observe().robot_state[5])
    for point in path:
        delta = point - previous
        if np.linalg.norm(delta) > 1.0e-6:
            yaw = math.atan2(delta[1], delta[0])
        env.set_robot_pose(float(point[0]), float(point[1]), yaw)
        viewer.sync()
        if not viewer.is_running():
            return False
        time.sleep(env.model.opt.timestep + 0.04 / speed)
        previous = point
    return True


def animate_push(
    env: BlockedPassageEnv,
    viewer: mujoco.viewer.Handle,
    target: np.ndarray,
    speed: float,
) -> bool:
    observation = env.observe()
    box = next(obj for obj in observation.objects if obj.movable)
    start = box.center[:2].copy()
    delta = target[:2] - start
    distance = float(np.linalg.norm(delta))
    direction = delta / distance
    yaw = math.atan2(direction[1], direction[0])
    robot_offset = box.size[0] / 2.0 + 0.34
    steps = max(2, math.ceil(distance / 0.03))
    for alpha in np.linspace(0.0, 1.0, steps):
        box_xy = start + alpha * delta
        robot_xy = box_xy - direction * robot_offset
        env.set_box_pose(float(box_xy[0]), float(box_xy[1]), yaw)
        env.set_robot_pose(float(robot_xy[0]), float(robot_xy[1]), yaw)
        viewer.sync()
        if not viewer.is_running():
            return False
        time.sleep(0.03 / speed)
    return True


def run_episode(
    env: BlockedPassageEnv,
    planner: OraclePlanner,
    viewer: mujoco.viewer.Handle,
    seed: int,
    speed: float,
) -> bool:
    observation = env.reset(seed=seed)
    plan = planner.plan(observation, env.push_target)
    if plan.reason != "reconfiguration_required":
        raise RuntimeError(f"Unexpected oracle result: {plan.reason}")

    print(f"[seed {seed}] BLOCKED: direct A* path does not exist")
    print("PLAN: NAV -> PUSH(object=10) -> NAV -> STOP")
    viewer.sync()
    if not pause(viewer, 1.5 / speed):
        return False

    approach_target = plan.actions[0].target_pose[:2]
    planner.grid.rasterize(
        observation.objects, planner.robot_radius(observation)
    )
    approach_path = planner.grid.astar(
        observation.robot_state[:2], approach_target
    )
    if approach_path is None:
        raise RuntimeError("Oracle approach path unexpectedly disappeared")
    print("EXECUTE NAV: approaching the movable box")
    if not animate_robot_path(env, viewer, approach_path, speed):
        return False

    print("EXECUTE PUSH: orange box moves beyond the goal")
    if not animate_push(env, viewer, env.push_target, speed):
        return False

    post_push = env.observe()
    goal_path = planner.path(post_push)
    if goal_path is None:
        raise RuntimeError("Goal should be reachable after reconfiguration")
    print("REPLAN: goal is reachable; executing final NAV")
    if not animate_robot_path(env, viewer, goal_path, speed):
        return False

    goal = post_push.goal
    env.set_robot_pose(float(goal[0]), float(goal[1]), 0.0)
    print("SUCCESS: robot reached the green goal marker")
    return pause(viewer, 2.0 / speed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--once", action="store_true", help="Close after one visualization cycle"
    )
    args = parser.parse_args()
    if args.speed <= 0.0:
        parser.error("--speed must be positive")

    env = BlockedPassageEnv()
    planner = OraclePlanner()
    env.reset(seed=args.seed)
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = [0.0, 0.0, 0.0]
        viewer.cam.distance = 6.2
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -55.0

        seed = args.seed
        while viewer.is_running():
            if not run_episode(env, planner, viewer, seed, args.speed):
                break
            if args.once:
                break
            seed += 1


if __name__ == "__main__":
    main()
