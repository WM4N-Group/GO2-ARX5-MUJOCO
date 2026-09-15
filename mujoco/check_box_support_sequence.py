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
from reconfigurable_navigation.box_navigation_runtime import BoxNavigationRuntime
from reconfigurable_navigation.box_robot_profile import box_runtime_provenance
from reconfigurable_navigation.locomotion_runtime import JOINT_NAMES, MJ_POLICY_INDICES, POLICY_MJ_INDICES, projected_gravity


def clear_arm(push):
    start_time = push.data.time
    point = push.arm.hand_position().copy()
    point[0] -= 0.12
    point[2] = push.data.geom_xpos[push.arm.box_geom, 2] + push.arm.box_size[2] / 2.0 + 0.15
    reason = "arm_clearance_timeout"
    for _step in range(round(4.0 / push.control_dt)):
        action = push.arm.apply(np.zeros(12), target=point)
        push._simulate_target(push.default_qpos + action[POLICY_MJ_INDICES] * push.action_scale)
        contact = push.arm.contacts()
        if contact["forbidden"] or contact["invalid_hand"] or push.base_contacts_terrain():
            reason = "arm_clearance_contact"
            break
        if push.data.qpos[2] < 0.18 or projected_gravity(push.data.qpos[3:7])[2] > -np.cos(0.9):
            reason = "arm_clearance_posture"
            break
        if np.linalg.norm(push.arm.hand_position() - point) < 0.035:
            reason = "arm_clear"
            break
    return {"skill": "CLEAR_ARM", "succeeded": reason == "arm_clear", "reason": reason, "elapsed": float(push.data.time - start_time), "target": point.tolist(), "hand_position": push.arm.hand_position().tolist()}


def climb_surface(runtime, surface, box, duration=12.0, *, climb_speed=0.30, heading_gain=0.0):
    model, data = runtime.model, runtime.data
    feet = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in ("FL", "FR", "RL", "RR")]
    surfaces = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in ("floor", "push_box", "high_platform")}
    start_time = data.time
    stable_steps = low_steps = max_supported = 0
    supported_stable_steps = 0
    navigation = None
    controller_transitions = []
    reason = "timeout"
    trace = []
    for step in range(round(duration / runtime.control_dt)):
        position = data.xpos[runtime.base_id]
        goal = data.geom_xpos[surface, :2]
        rotation = data.xmat[runtime.base_id].reshape(3, 3)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        delta = goal - position[:2]
        relative = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]]) @ delta
        velocity = np.array([np.clip(0.8 * relative[0], -0.15, climb_speed), np.clip(0.8 * relative[1], -0.12, 0.12), 0.0])
        surface_rotation = data.geom_xmat[surface].reshape(3, 3)
        target_yaw = np.arctan2(surface_rotation[1, 0], surface_rotation[0, 0])
        heading_error = np.arctan2(np.sin(target_yaw - yaw), np.cos(target_yaw - yaw))
        velocity[2] = np.clip(heading_gain * heading_error, -0.30, 0.30)
        if np.linalg.norm(delta) < 0.10:
            velocity.fill(0.0)
        if navigation is None:
            runtime.step(velocity)
        else:
            velocity[:2] = np.clip(1.5 * relative, -0.20, 0.20)
            for axis in (0, 1):
                if abs(relative[axis]) > 0.04 and abs(velocity[axis]) < 0.15:
                    velocity[axis] = np.copysign(0.15, velocity[axis])
            if np.linalg.norm(delta) < 0.10:
                velocity.fill(0.0)
            navigation.step(velocity)
            runtime.last_action = navigation.last_action.copy()
        base_contact = False
        for index in range(data.ncon):
            contact = data.contact[index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if pair & runtime.base_geom_ids and pair & surfaces:
                base_contact = True
        supported = runtime.support_contacts(feet, {surface})
        support_heights = list(runtime.support_contacts(feet, surfaces).values())
        max_supported = max(max_supported, len(supported))
        position = data.xpos[runtime.base_id]
        gravity = projected_gravity(data.xquat[runtime.base_id])
        clearance = position[2] - min(support_heights, default=0.0)
        low_steps = low_steps + 1 if clearance < 0.18 else 0
        if step % 25 == 0:
            trace.append({"time": float(data.time), "robot_position": position.tolist(), "box_position": data.geom_xpos[box].tolist(), "supported_feet": len(supported), "clearance": float(clearance)})
        surface_rotation = data.geom_xmat[surface].reshape(3, 3)
        local_feet = (data.geom_xpos[feet] - data.geom_xpos[surface]) @ surface_rotation
        safe_top = np.all(np.abs(local_feet[:, :2]) < model.geom_size[surface, :2] - 0.04) and np.all(np.abs(local_feet[:, 2] - model.geom_size[surface, 2] - 0.022) < 0.04)
        if base_contact:
            reason = "base_contact"
        elif gravity[2] > -np.cos(0.9):
            reason = "bad_orientation"
        elif data.geom_xmat[box].reshape(3, 3)[2, 2] < 0.94:
            reason = "box_tipped"
        elif low_steps * runtime.control_dt >= 0.2:
            reason = "low_posture"
        else:
            stable = len(supported) == 4 and safe_top and np.linalg.norm(position[:2] - data.geom_xpos[surface, :2]) < 0.20 and np.linalg.norm(data.qvel[:3]) < 0.15 and np.linalg.norm(data.qvel[3:6]) < 0.4 and gravity[2] < -0.94
            stable_steps = stable_steps + 1 if stable else 0
            inside_support = False
            if len(supported) >= 3:
                vertices = data.geom_xpos[list(supported), :2]
                centered = vertices - vertices.mean(axis=0)
                vertices = vertices[np.argsort(np.arctan2(centered[:, 1], centered[:, 0]))]
                edges = np.roll(vertices, -1, axis=0) - vertices
                offsets = data.subtree_com[runtime.base_id, :2] - vertices
                distances = (edges[:, 0] * offsets[:, 1] - edges[:, 1] * offsets[:, 0]) / np.maximum(np.linalg.norm(edges, axis=1), 1e-9)
                inside_support = bool(np.all(distances > 0.005))
            supported_stable = inside_support and safe_top and np.linalg.norm(data.qvel[:3]) < 0.15 and np.linalg.norm(data.qvel[3:6]) < 0.4 and gravity[2] < -0.94
            supported_stable_steps = supported_stable_steps + 1 if supported_stable else 0
            if navigation is None and not stable and supported_stable_steps * runtime.control_dt >= 0.30:
                navigation = BoxNavigationRuntime(runtime, blend_seconds=0.5)
                controller_transitions.append({"time": float(data.time), "controller": "NAV", "reason": "stable_top_position_alignment", "supported_feet": len(supported), "support_margin": float(distances.min())})
            if stable_steps * runtime.control_dt >= 1.0:
                reason = "surface_settled"
        if reason != "timeout":
            break
    terminal_contacts = [[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)), mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2))] for contact in data.contact[:data.ncon] if {int(contact.geom1), int(contact.geom2)} & runtime.base_geom_ids]
    return {"succeeded": reason == "surface_settled", "reason": reason, "elapsed": float(data.time - start_time), "max_supported_feet": max_supported, "controller_transitions": controller_transitions, "terminal_robot_position": data.xpos[runtime.base_id].tolist(), "terminal_gravity": projected_gravity(data.qpos[3:7]).tolist(), "terminal_base_contacts": terminal_contacts, "trace": trace}


def surface_entry(model, data, surface):
    rotation = data.geom_xmat[surface].reshape(3, 3)
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    forward = np.array([np.cos(yaw), np.sin(yaw)])
    position = data.geom_xpos[surface, :2] - (model.geom_size[surface, 0] + 0.75) * forward
    return position, yaw


def position_for_climb(runtime, box, *, support_height=0.0, blend_seconds=0.0):
    start_time = runtime.data.time
    reason = "approach_timeout"
    navigation = BoxNavigationRuntime(runtime, blend_seconds=blend_seconds)
    for _step in range(round(8.0 / runtime.control_dt)):
        position = runtime.data.xpos[runtime.base_id]
        target, target_yaw = surface_entry(runtime.model, runtime.data, box)
        delta = target - position[:2]
        rotation = runtime.data.xmat[runtime.base_id].reshape(3, 3)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        heading_error = np.arctan2(np.sin(target_yaw - yaw), np.cos(target_yaw - yaw))
        relative = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]]) @ delta
        velocity = np.array([np.clip(1.5 * relative[0], -0.20, 0.20), np.clip(1.5 * relative[1], -0.15, 0.15), np.clip(heading_error, -0.3, 0.3)])
        if abs(heading_error) > 0.06:
            velocity[2] = np.copysign(0.30, heading_error)
        for axis in (0, 1):
            minimum_speed = 0.25 if axis == 0 and relative[axis] < 0.0 else 0.15
            if abs(relative[axis]) > 0.04 and 0.0 < abs(velocity[axis]) < minimum_speed:
                velocity[axis] = np.copysign(minimum_speed, velocity[axis])
        navigation.step(velocity)
        runtime.last_action = navigation.last_action.copy()
        if runtime.base_contacts_terrain() or runtime.data.qpos[2] - support_height < 0.18 or projected_gravity(runtime.data.qpos[3:7])[2] > -np.cos(0.9):
            reason = "approach_posture"
            break
        if np.linalg.norm(delta) < 0.06 and abs(heading_error) < 0.08 and np.linalg.norm(runtime.data.qvel[:3]) < 0.15:
            reason = "approach_ready"
            break
    return {"skill": "POSITION_CLIMB", "succeeded": reason == "approach_ready", "reason": reason, "elapsed": float(runtime.data.time - start_time), "budget_seconds": 8.0, "position_error": float(np.linalg.norm(target - runtime.data.xpos[runtime.base_id, :2])), "speed": float(np.linalg.norm(runtime.data.qvel[:3])), "robot_position": runtime.data.xpos[runtime.base_id].tolist()}


def prepare_climb(climb, push, *, support_surface=None):
    model, data = climb.model, climb.data
    start_time = data.time
    support_height = 0.0 if support_surface is None else data.geom_xpos[support_surface, 2] + model.geom_size[support_surface, 2]
    feet = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in ("FL", "FR", "RL", "RR")}
    wrench = np.zeros(6)
    reference = data.qpos[climb.joint_qpos_adr[12:]].copy()
    folded = np.clip(climb.default_qpos[12:], push.arm.lower, push.arm.upper)
    reason = "preparation_timeout"
    stable_steps = 0
    support_geoms = {support_surface} if support_surface is not None else {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")}
    for _step in range(round(4.0 / climb.control_dt)):
        delta = folded - reference
        reference += delta * min(1.0, climb.control_dt / max(np.max(np.abs(delta)), 1.0e-9))
        push.arm.gravity_data.qpos[:] = data.qpos
        push.arm.gravity_data.qvel.fill(0.0)
        mujoco.mj_forward(model, push.arm.gravity_data)
        target = climb.default_qpos.copy()
        target[12:] = reference + push.arm.gravity_data.qfrc_bias[climb.joint_dof_adr[12:]] / climb.kp[12:]
        climb._simulate_target(target)
        climb.last_action = ((target - climb.default_qpos) / climb.action_scale)[MJ_POLICY_INDICES]
        forbidden = False
        for index in range(data.ncon):
            contact = data.contact[index]
            if push.arm.box_geom not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == push.arm.box_geom else contact.geom1
            if model.geom_bodyid[other] == 0 or support_surface is not None and other in feet:
                continue
            mujoco.mj_contactForce(model, data, index, wrench)
            forbidden |= wrench[0] > 1.0
        if climb.base_contacts_terrain() or forbidden:
            reason = "preparation_contact"
            break
        if data.qpos[2] - support_height < 0.18 or projected_gravity(data.qpos[3:7])[2] > -np.cos(0.9):
            reason = "preparation_posture"
            break
        settled = (
            np.max(np.abs(data.qpos[climb.joint_qpos_adr[12:]] - folded)) < 0.10
            and np.max(np.abs(data.qvel[climb.joint_dof_adr[:12]])) < 0.25
            and np.linalg.norm(data.qvel[:3]) < 0.05
            and np.linalg.norm(data.qvel[3:6]) < 0.10
            and len(climb.support_contacts(feet, support_geoms)) == 4
        )
        stable_steps = stable_steps + 1 if settled else 0
        if data.time - start_time > 0.5 and stable_steps * climb.control_dt >= 0.30:
            reason = "arm_retracted"
            climb.last_action.fill(0.0)
            break
    destination = push.arm.box_geom if support_surface is None else mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "high_platform")
    entry, entry_yaw = surface_entry(model, data, destination)
    origin = np.array([entry[0], entry[1], support_height])
    world_to_entry = np.array([[np.cos(entry_yaw), np.sin(entry_yaw), 0.0], [-np.sin(entry_yaw), np.cos(entry_yaw), 0.0], [0.0, 0.0, 1.0]])
    base_pose = data.qpos[:7].copy()
    base_pose[:3] = world_to_entry @ (base_pose[:3] - origin)
    mujoco.mju_mulQuat(base_pose[3:7], np.array([np.cos(entry_yaw / 2.0), 0.0, 0.0, -np.sin(entry_yaw / 2.0)]), data.qpos[3:7])
    velocity = data.qvel[:6].copy()
    velocity[:3] = world_to_entry @ velocity[:3]
    velocity[3:] = world_to_entry @ data.xmat[climb.base_id].reshape(3, 3) @ velocity[3:]
    return {
        "skill": "PREPARE_CLIMB", "succeeded": reason == "arm_retracted", "reason": reason,
        "elapsed": float(data.time - start_time), "policy_history_reset": reason == "arm_retracted",
        "stable_seconds": stable_steps * climb.control_dt,
        "prepared_state": {
            "phase": "ground" if support_surface is None else "platform",
            "reference_frame": "target_surface_yaw",
            "joint_names": [name.replace("x5_joint", "joint") for name in JOINT_NAMES],
            "base_pose": base_pose.tolist(), "base_velocity_world": velocity.tolist(),
            "joint_position": data.qpos[climb.joint_qpos_adr].tolist(),
            "joint_velocity": data.qvel[climb.joint_dof_adr].tolist(),
        },
    }


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