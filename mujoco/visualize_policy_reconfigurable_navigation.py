"""Run the physical NAV-to-PUSH-to-NAV pipeline in MuJoCo."""

from __future__ import annotations

import argparse
import time
from typing import Any

import mujoco
import mujoco.viewer

from reconfigurable_navigation.env import BlockedPassageEnv
from reconfigurable_navigation.locomotion_runtime import LocomotionRuntime
from reconfigurable_navigation.runtime import ReconfigurableExecutor


def sync(viewer: Any | None, control_dt: float, realtime: bool) -> bool:
    if viewer is not None:
        if not viewer.is_running():
            return False
        viewer.sync()
    if realtime:
        time.sleep(control_dt)
    return True


def run_pipeline(
    env: BlockedPassageEnv,
    runtime: LocomotionRuntime,
    viewer: Any | None,
    realtime: bool,
) -> bool:
    executor = ReconfigurableExecutor(env, runtime)

    def on_step(_action: object, _skill: object, _observation: object) -> bool:
        return sync(viewer, runtime.control_dt, realtime)

    result = executor.run(on_step=on_step, on_event=print)
    if not result.succeeded:
        print(
            f"FAILED: reason={result.reason} replans={result.replans} "
            f"skill_failures={result.skill_failures}"
        )
        return False

    observation = env.observe()
    box = next(obj for obj in observation.objects if obj.object_id == 10)
    print(
        "SUCCESS: receding-horizon physical task completed; "
        f"replans={result.replans}, "
        f"robot=({observation.robot_state[0]:.2f}, {observation.robot_state[1]:.2f}), "
        f"box=({box.center[0]:.2f}, {box.center[1]:.2f})"
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-realtime", action="store_true")
    args = parser.parse_args()

    env = BlockedPassageEnv()
    env.reset(seed=args.seed)
    runtime = LocomotionRuntime(env)
    if args.headless:
        success = run_pipeline(env, runtime, None, not args.no_realtime)
    else:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = runtime.base_id
            viewer.cam.distance = 3.5
            viewer.cam.azimuth = 145.0
            viewer.cam.elevation = -25.0
            success = run_pipeline(env, runtime, viewer, not args.no_realtime)
            if viewer.is_running():
                print("Close the Viewer window to exit.")
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(1.0 / 60.0)
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()