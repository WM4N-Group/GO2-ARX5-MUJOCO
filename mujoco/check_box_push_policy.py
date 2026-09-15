"""Strict physical check of the experimental MuJoCo hybrid PUSH runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from reconfigurable_navigation.box_push_runtime import BoxPushRuntime
from reconfigurable_navigation.box_robot_profile import apply_box_robot_profile, box_runtime_provenance
from reconfigurable_navigation.locomotion_runtime import JOINT_NAMES, projected_gravity


ROOT = Path(__file__).resolve().parent


def make_model(height=0.20, mass=5.0, friction=0.4, platform_height=None):
    if not all(np.isfinite(value) and value > 0.0 for value in (height, mass, friction)):
        raise ValueError("Physical parameters must be finite and positive")
    spec = mujoco.MjSpec.from_file(str(ROOT / "robots/go2_arx5/go2_arx5.xml"))
    apply_box_robot_profile(spec)
    spec.worldbody.add_light(pos=[0.0, -3.0, 5.0], dir=[0.0, 0.0, -1.0], diffuse=[0.8, 0.8, 0.8], ambient=[0.35, 0.35, 0.35])
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0.0, 0.0, 0.05], friction=[0.6, 0.0, 0.0], priority=1, condim=3, conaffinity=3)
    box = spec.worldbody.add_body(name="push_box", pos=[1.10, 0.0, height / 2.0])
    box.add_freejoint(name="push_box_joint")
    box.add_geom(name="push_box", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.6, 0.6, height / 2.0], mass=mass, friction=[friction, 0.0, 0.0], priority=2, condim=3, conaffinity=3, rgba=[0.65, 0.3, 0.15, 1.0])
    if platform_height is not None:
        if not np.isfinite(platform_height) or platform_height <= height:
            raise ValueError("Platform must be higher than the box")
        spec.worldbody.add_geom(name="high_platform", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[3.3, 0.0, platform_height / 2.0], size=[1.0, 0.8, platform_height / 2.0], friction=[0.6, 0.0, 0.0], priority=1, condim=3, conaffinity=3, rgba=[0.25, 0.55, 0.72, 1.0])
    model = spec.compile()
    for name in JOINT_NAMES:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        dof = model.jnt_dofadr[joint]
        model.dof_damping[dof] = 0.0
        model.dof_frictionloss[dof] = 0.01
        model.dof_armature[dof] = 0.01
    return model


def make_episode(policy, seed, platform_height=None):
    model = make_model(platform_height=platform_height)
    data = mujoco.MjData(model)
    box = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "push_box")
    runtime = BoxPushRuntime(policy, model=model, data=data, box_geom=box, goal=np.array([1.70, 0.0, 0.10]))
    generator = np.random.default_rng(seed)
    data.qpos[:3] = [*generator.uniform(-0.03, 0.03, size=2), 0.33]
    yaw = generator.uniform(-0.04, 0.04)
    data.qpos[3:7] = [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]
    joint_ids = model.dof_jntid[runtime.joint_dof_adr]
    midpoint = model.jnt_range[joint_ids].mean(axis=1)
    half_range = 0.45 * np.ptp(model.jnt_range[joint_ids], axis=1)
    data.qpos[runtime.joint_qpos_adr] = np.clip(runtime.default_qpos * generator.uniform(0.95, 1.05, 18), midpoint - half_range, midpoint + half_range)
    box_qpos = int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "push_box_joint")])
    data.qpos[box_qpos:box_qpos + 2] += generator.uniform(-0.03, 0.03, 2)
    yaw = generator.uniform(-0.04, 0.04)
    data.qpos[box_qpos + 3:box_qpos + 7] = [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]
    mujoco.mj_forward(model, data)
    start = data.geom_xpos[box].copy()
    runtime.arm.goal = start + [0.60, 0.0, 0.0]
    runtime.activate()
    return runtime, start


def run_episode(policy, seed, duration=12.0, *, runtime=None):
    if runtime is None:
        runtime, start = make_episode(policy, seed)
    else:
        start = runtime.data.geom_xpos[runtime.arm.box_geom].copy()
    model, data, box = runtime.model, runtime.data, runtime.arm.box_geom
    start_time = data.time
    stable_steps = low_steps = valid_steps = 0
    reason = "timeout"
    trace = []
    for step in range(round(duration / runtime.control_dt)):
        runtime.step()
        contact = runtime.arm.contacts()
        valid_steps += int(contact["valid"])
        position = data.xpos[runtime.base_id].copy()
        gravity = projected_gravity(data.xquat[runtime.base_id])
        low_steps = low_steps + 1 if data.time >= 0.4 and position[2] < 0.18 else 0
        error = np.linalg.norm(runtime.arm.goal[:2] - data.geom_xpos[box, :2])
        speed = np.linalg.norm(runtime._box_velocity())
        if step % 25 == 0:
            trace.append({"time": float(data.time), "robot_position": position.tolist(), "box_position": data.geom_xpos[box].tolist(), "hand_position": runtime.arm.hand_position().tolist(), "hand_force": contact["force"].tolist(), "locked": bool(runtime.arm.locked), "goal_error": float(error)})
        if contact["forbidden"]:
            reason = "box_contact"
        elif contact["invalid_hand"]:
            reason = "invalid_hand_contact"
        elif runtime.base_contacts_terrain():
            reason = "base_contact"
        elif gravity[2] > -np.cos(0.9):
            reason = "bad_orientation"
        elif runtime.arm.box_rotation()[2, 2] < 0.94:
            reason = "box_tipped"
        elif low_steps * runtime.control_dt >= 0.2:
            reason = "low_posture"
        else:
            stable = error < 0.12 and speed < 0.08 and valid_steps > 0
            stable_steps = stable_steps + 1 if stable else 0
            if stable_steps * runtime.control_dt >= 0.5:
                reason = "box_settled"
        if reason != "timeout":
            break
    result = {"seed": seed, "succeeded": reason == "box_settled", "reason": reason, "elapsed": float(data.time - start_time), "valid_contact_steps": valid_steps, "box_displacement": (data.geom_xpos[box] - start).tolist(), "goal_error": float(error), "box_speed": float(speed), "trace": trace}
    print(f"BOX_PUSH seed={seed} reason={reason} error={error:.3f} contact_steps={valid_steps}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.seeds < 1 or args.seed_offset < 0 or args.output_json.exists():
        parser.error("Use positive seed count, nonnegative offset and a new output file")
    torch.set_num_threads(1)
    results = [run_episode(args.policy, seed) for seed in range(args.seed_offset, args.seed_offset + args.seeds)]
    successes = sum(result["succeeded"] for result in results)
    passed = successes / len(results) >= 29 / 32
    paths = [Path(__file__), ROOT / "reconfigurable_navigation/box_push_runtime.py", ROOT / "reconfigurable_navigation/climb_runtime.py"]
    report = {"schema_version": 1, "policy_sha256": hashlib.sha256(args.policy.read_bytes()).hexdigest(), "code_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}, "mujoco_version": mujoco.__version__, "fixed_support": False, "box_size": [1.2, 1.2, 0.20], "box_mass": 5.0, "box_friction": 0.4, "push_distance": 0.60, "passed": passed, "results": results}
    report["runtime_provenance"] = box_runtime_provenance()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(f"MuJoCo hybrid PUSH: {successes}/{len(results)}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()