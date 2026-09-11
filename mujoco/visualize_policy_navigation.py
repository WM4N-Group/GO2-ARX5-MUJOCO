"""Run policy-driven NAV in the blocked-passage MuJoCo scene."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np

from reconfigurable_navigation.env import BlockedPassageEnv
from reconfigurable_navigation.locomotion_runtime import LocomotionRuntime
from reconfigurable_navigation.oracle_planner import OraclePlanner
from reconfigurable_navigation.skills import NavigateSkill, SkillStatus


def sync(viewer: Any | None, control_dt: float, realtime: bool) -> bool:
    if viewer is not None:
        if not viewer.is_running():
            return False
        viewer.sync()
    if realtime:
        time.sleep(control_dt)
    return True


def run_navigation(
    env: BlockedPassageEnv,
    runtime: LocomotionRuntime,
    viewer: Any | None,
    timeout: float,
    realtime: bool,
) -> bool:
    observation = env.observe()
    plan = OraclePlanner().plan(observation, env.push_target)
    if plan.reason != "reconfiguration_required":
        raise RuntimeError(f"Unexpected oracle result: {plan.reason}")
    nav_action = plan.actions[0]
    skill = NavigateSkill()
    skill.reset(nav_action)

    print("STABILIZE: holding the nominal joint pose for 3 seconds")
    for _ in range(round(3.0 / runtime.control_dt)):
        runtime.hold_default()
        if not sync(viewer, runtime.control_dt, realtime):
            return False

    start_xy = env.observe().robot_state[:2].copy()
    target_xy = nav_action.target_pose[:2]
    print(
        "EXECUTE NAV: policy-driven locomotion from "
        f"({start_xy[0]:.2f}, {start_xy[1]:.2f}) to "
        f"({target_xy[0]:.2f}, {target_xy[1]:.2f})"
    )
    max_steps = round(timeout / runtime.control_dt)
    report_interval = max(1, round(1.0 / runtime.control_dt))
    for step in range(max_steps):
        observation = env.observe()
        command = skill.step(observation)
        runtime.step(command.velocity, command.end_effector_pose)
        if not sync(viewer, runtime.control_dt, realtime):
            return False

        if step % report_interval == 0:
            position = env.observe().robot_state[:3]
            distance = float(np.linalg.norm(target_xy - position[:2]))
            print(
                f"  t={step * runtime.control_dt:5.1f}s "
                f"position=({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}) "
                f"distance={distance:.2f}m"
            )
        if skill.status == SkillStatus.SUCCEEDED:
            position = env.observe().robot_state[:3]
            print(
                "NAV SUCCESS: actual MuJoCo dynamics reached "
                f"({position[0]:.2f}, {position[1]:.2f})"
            )
            return True
        if skill.status == SkillStatus.FAILED:
            print("NAV FAILED: skill reported invalid state or timeout")
            return False
        if env.observe().robot_state[2] < 0.18:
            print("NAV FAILED: robot base height indicates a fall")
            return False

    position = env.observe().robot_state[:3]
    distance = float(np.linalg.norm(target_xy - position[:2]))
    print(
        f"NAV FAILED: timeout at ({position[0]:.2f}, {position[1]:.2f}), "
        f"remaining distance={distance:.2f}m"
    )
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--action-clip", type=float, default=20.0)
    parser.add_argument(
        "--policy",
        type=Path,
        default=None,
        help="Override the default GO2-ARX5 TorchScript policy",
    )
    parser.add_argument(
        "--headless", action="store_true", help="Run without a MuJoCo Viewer"
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Do not sleep between control steps",
    )
    args = parser.parse_args()
    if args.timeout <= 0.0:
        parser.error("--timeout must be positive")
    if args.action_clip <= 0.0:
        parser.error("--action-clip must be positive")

    env = BlockedPassageEnv()
    env.reset(seed=args.seed)
    runtime = (
        LocomotionRuntime(env, args.policy, args.action_clip)
        if args.policy is not None
        else LocomotionRuntime(env, action_clip=args.action_clip)
    )
    if args.headless:
        success = run_navigation(
            env, runtime, None, args.timeout, not args.no_realtime
        )
    else:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = runtime.base_id
            viewer.cam.distance = 3.5
            viewer.cam.azimuth = 145.0
            viewer.cam.elevation = -25.0
            success = run_navigation(
                env, runtime, viewer, args.timeout, not args.no_realtime
            )
            if viewer.is_running():
                print("Close the Viewer window to exit.")
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(1.0 / 60.0)
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
