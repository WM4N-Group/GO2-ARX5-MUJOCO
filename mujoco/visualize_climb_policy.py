"""Visualize the GO2-ARX5 CLIMB policy on low MuJoCo stairs."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import mujoco
import mujoco.viewer

from reconfigurable_navigation.climb_runtime import ClimbRuntime, POLICY_PATH


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, default=POLICY_PATH)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--velocity", type=float, default=0.5)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-realtime", action="store_true")
    args = parser.parse_args()
    if args.duration <= 0.0:
        parser.error("--duration must be positive")

    runtime = ClimbRuntime(args.policy)
    runtime.velocity_command[0] = args.velocity
    viewer = None
    if not args.headless:
        viewer = mujoco.viewer.launch_passive(runtime.model, runtime.data)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = runtime.base_id
        viewer.cam.distance = 4.0
        viewer.cam.azimuth = 145.0
        viewer.cam.elevation = -20.0

    print(f"CLIMB DIAGNOSTIC: {args.policy.name}")
    start_x = float(runtime.data.xpos[runtime.base_id, 0])
    min_height = float(runtime.data.xpos[runtime.base_id, 2])
    fell = False
    steps = round(args.duration / runtime.control_dt)
    try:
        for step in range(steps):
            if viewer is not None and not viewer.is_running():
                break
            runtime.step()
            if viewer is not None:
                viewer.sync()
            if not args.no_realtime:
                time.sleep(runtime.control_dt)
            base_position = runtime.data.xpos[runtime.base_id]
            min_height = min(min_height, float(base_position[2]))
            if runtime.base_contacts_terrain():
                fell = True
                print(
                    "CLIMB FAILED: robot base contacted the terrain at "
                    f"z={base_position[2]:.2f}m"
                )
                break
            if step % round(1.0 / runtime.control_dt) == 0:
                print(
                    f"t={step * runtime.control_dt:5.1f}s "
                    f"base=({base_position[0]:.2f}, {base_position[1]:.2f}, "
                    f"{base_position[2]:.2f})"
                )
        base_position = runtime.data.xpos[runtime.base_id]
        print(
            "CLIMB DIAGNOSTIC COMPLETE: "
            f"x_displacement={base_position[0] - start_x:.2f}m "
            f"final_height={base_position[2]:.2f}m "
            f"min_height={min_height:.2f}m fell={fell}"
        )
        if viewer is not None and viewer.is_running():
            print("Close the Viewer window to exit.")
            while viewer.is_running():
                viewer.sync()
                time.sleep(1.0 / 60.0)
    finally:
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    main()