"""Diagnostic PUSH and two-stage CLIMB on one continuously simulated world."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from check_box_push_policy import make_episode, run_episode
from reconfigurable_navigation.box_climb_runtime import BoxClimbRuntime
from reconfigurable_navigation.box_robot_profile import box_runtime_provenance
from reconfigurable_navigation.box_support_control import clear_arm, climb_surface, position_for_climb, prepare_climb, surface_entry


def run_sequence(push_policy, climb_policy, seed, *, platform_policy=None, on_step=None, climb_speed=0.30, heading_gain=0.0):
    push, _start = make_episode(push_policy, seed, platform_height=0.40)
    push.on_step = on_step
    model, data = push.model, push.data
    platform = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "high_platform")
    platform_edge = data.geom_xpos[platform, 0] - model.geom_size[platform, 0]
    placement_target = np.array([platform_edge - push.arm.box_size[0] / 2.0, data.geom_xpos[platform, 1], push.arm.goal[2]])
    push.arm.goal = placement_target + [0.02, 0.0, 0.0]
    push.activate()
    stages = [{"skill": "PUSH", **run_episode(push_policy, seed, runtime=push)}]
    box = push.arm.box_geom
    box_after_push = data.geom_xpos[box].copy()
    if stages[-1]["succeeded"]:
        stages.append(clear_arm(push))
    if stages[-1]["succeeded"]:
        climb = BoxClimbRuntime(climb_policy, model=model, data=data)
        climb.on_step = on_step
        climb.joint_velocity_limits[12:] = 3.0
        time_before = data.time
        robot_before = data.qpos.copy()
        climb.activate()
        np.testing.assert_array_equal(data.qpos, robot_before)
        if data.time != time_before:
            raise AssertionError("Skill activation changed physical time")
        climb.inherit_actuator_state(push)
        stages.append(position_for_climb(climb, box))
        if stages[-1]["succeeded"]:
            stages.append(prepare_climb(climb, push))
        if stages[-1]["succeeded"]:
            stages.append({"skill": "CLIMB_BOX", **climb_surface(climb, box, box, climb_speed=climb_speed, heading_gain=heading_gain)})
        if stages[-1]["succeeded"]:
            stages.append({**position_for_climb(climb, platform, support_height=0.20, blend_seconds=0.5), "skill": "POSITION_PLATFORM"})
        if stages[-1]["succeeded"]:
            platform_runtime = BoxClimbRuntime(platform_policy or climb_policy, model=model, data=data)
            platform_runtime.joint_velocity_limits[12:] = 3.0
            platform_runtime.on_step = on_step
            platform_runtime.inherit_actuator_state(climb)
            stages.append({**prepare_climb(platform_runtime, push, support_surface=box), "skill": "PREPARE_PLATFORM"})
        if stages[-1]["succeeded"]:
            stages.append({"skill": "CLIMB_PLATFORM", **climb_surface(platform_runtime, platform, box, climb_speed=climb_speed, heading_gain=heading_gain)})
    result = {
        "seed": seed,
        "succeeded": len(stages) == 8 and all(stage["succeeded"] for stage in stages),
        "stages": stages,
        "physical_initializations": 1,
        "skill_switch_resets": 0,
        "elapsed": float(data.time),
        "box_motion_after_push": (data.geom_xpos[box] - box_after_push).tolist(),
        "final_robot_position": data.xpos[push.base_id].tolist(),
        "placement_target": placement_target.tolist(),
        "push_control_target": push.arm.goal.tolist(),
        "placement_gap": float(platform_edge - box_after_push[0] - push.arm.box_size[0] / 2.0),
    }
    print(f"SEQUENCE seed={seed} stages={[(stage['skill'], stage['reason']) for stage in stages]}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--push-policy", type=Path, required=True)
    parser.add_argument("--climb-policy", type=Path, required=True)
    parser.add_argument("--platform-policy", type=Path)
    parser.add_argument("--climb-speed", type=float, default=0.30)
    parser.add_argument("--heading-gain", type=float, default=0.0)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--video-path", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if not np.isfinite(args.climb_speed) or args.climb_speed <= 0.0:
        parser.error("Climb speed must be positive and finite")
    if not np.isfinite(args.heading_gain) or args.heading_gain < 0.0:
        parser.error("Heading gain must be nonnegative and finite")
    if args.seeds < 1 or args.seed_offset < 0 or args.output_json.exists():
        parser.error("Use a positive seed count and a new output file")
    if args.video_path is not None and (args.video_path.exists() or args.seeds != 1):
        parser.error("Video requires one episode and a new output file")
    torch.set_num_threads(1)
    renderer = None
    writer = None
    frames = 0
    control_steps = 0

    def record_frame(runtime):
        nonlocal renderer, frames, control_steps
        if renderer is None:
            runtime.model.vis.global_.offwidth = 1280
            runtime.model.vis.global_.offheight = 720
            renderer = mujoco.Renderer(runtime.model, height=720, width=1280)
        if control_steps % 2 == 0:
            camera = mujoco.MjvCamera()
            camera.lookat[:] = [1.8, 0.0, 0.25]
            camera.distance = 4.2
            camera.azimuth = 105.0
            camera.elevation = -22.0
            renderer.update_scene(runtime.data, camera)
            writer.append_data(renderer.render())
            frames += 1
        control_steps += 1

    try:
        if args.video_path is not None:
            import imageio.v2 as imageio

            args.video_path.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(str(args.video_path), fps=25, codec="libx264")
        results = [run_sequence(args.push_policy, args.climb_policy, seed, platform_policy=args.platform_policy, on_step=record_frame if writer is not None else None, climb_speed=args.climb_speed, heading_gain=args.heading_gain) for seed in range(args.seed_offset, args.seed_offset + args.seeds)]
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
    sources = [Path(__file__), Path(__file__).with_name("check_box_push_policy.py"), Path(__file__).parent / "reconfigurable_navigation/box_push_runtime.py", Path(__file__).parent / "reconfigurable_navigation/climb_runtime.py"]
    report = {"schema_version": 1, "diagnostic": True, "fixed_box": False, "box_mass": 5.0, "box_friction": 0.4, "box_height": 0.20, "platform_height": 0.40, "mujoco_version": mujoco.__version__, "policy_sha256": {"push": hashlib.sha256(args.push_policy.read_bytes()).hexdigest(), "climb": hashlib.sha256(args.climb_policy.read_bytes()).hexdigest()}, "code_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}, "results": results}
    report.update(video_path=str(args.video_path) if args.video_path is not None else None, video_frames=frames)
    report["climb_speed"] = args.climb_speed
    report["heading_gain"] = args.heading_gain
    report["runtime_provenance"] = box_runtime_provenance()
    report["policy_sha256"]["platform"] = hashlib.sha256((args.platform_policy or args.climb_policy).read_bytes()).hexdigest()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    raise SystemExit(0 if all(result["succeeded"] for result in results) else 1)


if __name__ == "__main__":
    main()