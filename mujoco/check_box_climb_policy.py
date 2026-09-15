"""Evaluate an experimental box-climb actor without replacing the baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from reconfigurable_navigation.box_climb_runtime import BoxClimbRuntime
from reconfigurable_navigation.box_robot_profile import apply_box_robot_profile, box_runtime_provenance
from reconfigurable_navigation.locomotion_runtime import projected_gravity


ROOT = Path(__file__).resolve().parent


def make_model(height: float, approach_height=0.0, landing_length=1.2, landing_width=1.2, approach_gap=0.0) -> mujoco.MjModel:
    if not np.isfinite(height) or height <= 0.0:
        raise ValueError("Box height must be positive and finite")
    spec = mujoco.MjSpec.from_file(str(ROOT / "robots/go2_arx5/go2_arx5.xml"))
    apply_box_robot_profile(spec)
    spec.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0.0, 0.0, 0.05],
        friction=[0.6, 0.0, 0.0], priority=2, condim=3, conaffinity=3,
    )
    spec.worldbody.add_geom(
        name="support_box", type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.75 + landing_length / 2.0, 0.0, (height + approach_height) / 2.0], size=[landing_length / 2.0, landing_width / 2.0, (height + approach_height) / 2.0],
        friction=[0.6, 0.0, 0.0], priority=2, condim=3, conaffinity=3, rgba=[0.25, 0.55, 0.72, 1.0],
    )
    if approach_height > 0.0:
        spec.worldbody.add_geom(name="approach_box", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.15 - approach_gap, 0.0, approach_height / 2.0], size=[0.6, 0.6, approach_height / 2.0], friction=[0.6, 0.0, 0.0], priority=2, condim=3, conaffinity=3)
    return spec.compile()


def run_episode(policy: Path, height: float, seed: int, *, approach_height=0.0, landing_length=1.2, landing_width=1.2, approach_gap=0.0) -> dict:
    model = make_model(height, approach_height, landing_length, landing_width, approach_gap)
    data = mujoco.MjData(model)
    runtime = BoxClimbRuntime(policy, model=model, data=data)
    runtime.joint_velocity_limits[12:] = 3.0
    model.dof_damping[runtime.joint_dof_adr] = 0.0
    model.dof_frictionloss[runtime.joint_dof_adr] = 0.01
    model.dof_armature[runtime.joint_dof_adr] = 0.01
    generator = np.random.default_rng(seed)
    mujoco.mj_resetData(model, data)
    data.qpos[:3] = [generator.uniform(-0.1, 0.1), generator.uniform(-0.08, 0.08), 0.33 + approach_height]
    yaw = generator.uniform(-0.06, 0.06)
    data.qpos[3:7] = [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]
    joint_ids = model.dof_jntid[runtime.joint_dof_adr]
    midpoint = model.jnt_range[joint_ids].mean(axis=1)
    half_range = 0.45 * (model.jnt_range[joint_ids, 1] - model.jnt_range[joint_ids, 0])
    data.qpos[runtime.joint_qpos_adr] = np.clip(
        runtime.default_qpos * generator.uniform(0.9, 1.1, size=18),
        midpoint - half_range, midpoint + half_range,
    )
    mujoco.mj_forward(model, data)
    runtime.activate()
    feet = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in ("FL", "FR", "RL", "RR")]
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    box = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "support_box")
    support_geoms = {floor, box}
    if approach_height > 0.0:
        support_geoms.add(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "approach_box"))
    goal = data.geom_xpos[box, :2].copy()
    stable_steps = 0
    low_steps = 0
    max_supported = 0
    reason = "timeout"
    trace = []
    for step in range(round(12.0 / runtime.control_dt)):
        position = data.xpos[runtime.base_id].copy()
        rotation = data.xmat[runtime.base_id].reshape(3, 3)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        delta = goal - position[:2]
        relative = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]]) @ delta
        velocity = np.array([np.clip(0.8 * relative[0], -0.15, 0.30), np.clip(0.8 * relative[1], -0.12, 0.12), 0.0])
        if np.linalg.norm(delta) < 0.10:
            velocity.fill(0.0)
        runtime.step(velocity)
        supported = runtime.support_contacts(feet, {box})
        surfaces = list(runtime.support_contacts(feet, support_geoms).values())
        max_supported = max(max_supported, len(supported))
        position = data.xpos[runtime.base_id].copy()
        gravity = projected_gravity(data.qpos[3:7])
        clearance = position[2] - min(surfaces, default=0.0)
        low_steps = low_steps + 1 if data.time >= 0.4 and clearance < 0.18 else 0
        if step % 25 == 0:
            trace.append({"time": float(data.time), "position": position.tolist(), "supported_feet": len(supported), "clearance": float(clearance)})
        if runtime.base_contacts_terrain():
            reason = "base_contact"
            break
        if gravity[2] > -np.cos(0.9):
            reason = "bad_orientation"
            break
        if low_steps * runtime.control_dt >= 0.2:
            reason = "low_posture"
            break
        feet_positions = data.geom_xpos[feet]
        on_top = np.all(np.abs(feet_positions[:, :2] - goal) < model.geom_size[box, :2] - 0.04) and np.all(np.abs(feet_positions[:, 2] - (height + approach_height + 0.022)) < 0.04)
        stable = len(supported) == 4 and on_top and np.linalg.norm(position[:2] - goal) < 0.20 and np.linalg.norm(data.qvel[:3]) < 0.15 and np.linalg.norm(data.qvel[3:6]) < 0.4 and gravity[2] < -0.94
        stable_steps = stable_steps + 1 if stable else 0
        if stable_steps * runtime.control_dt >= 1.0:
            reason = "box_settled"
            break
    result = {"seed": seed, "height": height, "succeeded": reason == "box_settled", "reason": reason, "elapsed": float(data.time), "max_supported_feet": max_supported, "trace": trace}
    print(f"BOX_CLIMB seed={seed} height={height} success={result['succeeded']} reason={reason}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--height", type=float, default=0.20)
    parser.add_argument("--approach-height", type=float, default=0.0)
    parser.add_argument("--approach-gap", type=float, default=0.0)
    parser.add_argument("--landing-length", type=float, default=1.2)
    parser.add_argument("--landing-width", type=float, default=1.2)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists() or args.seeds < 1 or args.seed_offset < 0:
        parser.error("Use a new output file and a positive seed count")
    if any(not np.isfinite(value) or value < 0.0 for value in (args.approach_height, args.approach_gap)) or min(args.landing_length, args.landing_width) <= 0.08:
        parser.error("Invalid approach or landing geometry")
    torch.set_num_threads(1)
    geometry = {name: getattr(args, name) for name in ("approach_height", "approach_gap", "landing_length", "landing_width")}
    results = [run_episode(args.policy, args.height, seed, **geometry) for seed in range(args.seed_offset, args.seed_offset + args.seeds)]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    successes = sum(result["succeeded"] for result in results)
    passed = successes / len(results) >= 0.90
    payload = {
        "policy_sha256": hashlib.sha256(args.policy.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "mujoco_version": mujoco.__version__, "fixed_support": True, "friction": 0.6,
        "reset_soft_limit_factor": 0.9,
        "runtime_provenance": box_runtime_provenance(),
        "geometry": geometry,
        "min_success_rate": 0.90, "passed": passed, "results": results,
    }
    with args.output_json.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    print(f"Box CLIMB: {successes}/{len(results)}; gate_passed={passed}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()