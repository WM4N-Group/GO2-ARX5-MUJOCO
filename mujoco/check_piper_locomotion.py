"""Run a headless GO2-PIPER locomotion smoke test with per-physics-step checks."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch

from check_piper_startup import sha256
from reconfigurable_navigation.piper_locomotion_runtime import DEPLOY, PiperLocomotionRuntime


class RolloutFailure(RuntimeError):
    pass


class PhysicsMonitor:
    def __init__(self, runtime, command):
        self.runtime = runtime
        self.command = command
        self.stage = "stabilize"
        self.min_height = float("inf")
        self.max_tilt = 0.0
        self.max_torque_fraction = 0.0
        self.physics_steps = 0
        self.last_time = float(runtime.data.time)
        self.first_illegal_contact = None
        self.velocities = []
        self.foot_geoms = {
            mujoco.mj_name2id(runtime.model, mujoco.mjtObj.mjOBJ_GEOM, leg)
            for leg in ("FL", "FR", "RL", "RR")
        }
        self.floor_geom = mujoco.mj_name2id(runtime.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if -1 in self.foot_geoms or self.floor_geom < 0:
            raise ValueError("Missing floor/foot geometries")

    def __call__(self):
        model, data = self.runtime.model, self.runtime.data
        self.physics_steps += 1
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or not np.isfinite(data.ctrl).all():
            raise RolloutFailure("nonfinite_state_or_control")
        if data.time <= self.last_time:
            raise RolloutFailure("unexpected_physics_reset")
        self.last_time = float(data.time)
        base_rotation = data.xmat[self.runtime.base_id].reshape(3, 3)
        height = float(data.qpos[self.runtime.env.robot_qpos_adr + 2])
        tilt = float(np.arccos(np.clip(base_rotation[2, 2], -1.0, 1.0)))
        self.min_height = min(self.min_height, height)
        self.max_tilt = max(self.max_tilt, tilt)
        limits = np.maximum(np.abs(self.runtime.ctrl_low), np.abs(self.runtime.ctrl_high))
        self.max_torque_fraction = max(self.max_torque_fraction, float(np.max(np.abs(data.ctrl) / limits)))
        if height < 0.16:
            raise RolloutFailure("low_base_height")
        if tilt > np.pi / 3:
            raise RolloutFailure("bad_orientation")
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            if contact.dist > 0 or self.floor_geom not in (contact.geom1, contact.geom2):
                continue
            other_geom = contact.geom2 if contact.geom1 == self.floor_geom else contact.geom1
            if other_geom in self.foot_geoms:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, contact_index, force)
            if force[0] > 1.0:
                self.first_illegal_contact = {
                    "time": float(data.time),
                    "body": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[other_geom])),
                    "normal_force": float(force[0]),
                    "depth": float(-contact.dist),
                }
                raise RolloutFailure("nonfoot_ground_contact")
        if self.stage == "policy":
            root_dof = self.runtime.robot_dof_adr
            local_velocity = base_rotation.T @ data.qvel[root_dof:root_dof + 3]
            self.velocities.append([*local_velocity[:2], data.qvel[root_dof + 5]])


def run_episode(args, seed):
    filename = "config_wbc.yaml" if args.policy == "wbc" else "config.yaml"
    runtime = PiperLocomotionRuntime(DEPLOY / filename, seed=seed, initialization=args.initialization, profile_path=args.robot_profile)
    command = np.asarray(args.velocity, dtype=np.float64)
    runtime.ee_command[:] = [*args.ee_position, 1.0, 0.0, 0.0, 0.0]
    runtime.reset()
    monitor = PhysicsMonitor(runtime, command)
    runtime.on_physics_step = monitor
    reason = "completed"
    start_position = runtime.data.qpos[:3].copy()
    try:
        for _ in range(round(args.settle / runtime.control_dt)):
            runtime.hold_default()
        start_position = runtime.data.qpos[:3].copy()
        monitor.stage = "policy"
        for _ in range(round(args.duration / runtime.control_dt)):
            runtime.step(command)
    except RolloutFailure as error:
        reason = str(error)
    velocities = np.asarray(monitor.velocities).reshape(-1, 3)
    mean_velocity = velocities.mean(axis=0) if len(velocities) else np.zeros(3)
    errors = velocities - command if len(velocities) else None
    velocity_rmse = np.sqrt(np.mean(errors ** 2, axis=0)) if errors is not None else None
    if reason == "completed" and (
        np.linalg.norm(mean_velocity[:2] - command[:2]) > max(0.12, 0.5 * np.linalg.norm(command[:2]))
        or abs(mean_velocity[2] - command[2]) > 0.2
    ):
        reason = "velocity_tracking"
    return {
        "seed": seed,
        "passed": reason == "completed",
        "reason": reason,
        "stage": monitor.stage,
        "elapsed_sim_time": float(runtime.data.time),
        "physics_steps": monitor.physics_steps,
        "physical_initializations": 1,
        "min_base_height": monitor.min_height,
        "max_tilt_rad": monitor.max_tilt,
        "max_torque_fraction": monitor.max_torque_fraction,
        "displacement_world": (runtime.data.qpos[:3] - start_position).tolist(),
        "mean_body_velocity": mean_velocity.tolist(),
        "body_velocity_rmse": velocity_rmse.tolist() if velocity_rmse is not None else None,
        "first_illegal_contact": monitor.first_illegal_contact,
        "policy_sha256": sha256(runtime.policy_path),
        "config_sha256": sha256(runtime.config_path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("flat", "wbc"), default="flat")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--settle", type=float, default=3.0)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--velocity", type=float, nargs=3, default=(0.25, 0.0, 0.0))
    parser.add_argument("--ee-position", type=float, nargs=3, default=(0.5, 0.0, 0.4))
    parser.add_argument("--initialization", choices=("nominal", "model"), default="nominal")
    parser.add_argument("--robot-profile", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if not np.isfinite([args.duration, args.settle, *args.velocity, *args.ee_position]).all():
        parser.error("All numeric parameters must be finite")
    if args.duration < 0.02 or args.settle < 0 or args.seeds < 1 or args.seed_offset < 0:
        parser.error("Require duration >= 0.02, settle >= 0, seeds >= 1 and seed-offset >= 0")
    if args.output_json is not None and args.output_json.exists():
        parser.error(f"Refusing to overwrite {args.output_json}")
    torch.set_num_threads(1)
    results = []
    for seed in range(args.seed_offset, args.seed_offset + args.seeds):
        result = run_episode(args, seed)
        results.append(result)
        print(f"{args.policy} seed={seed} passed={result['passed']} reason={result['reason']}", file=sys.stderr, flush=True)
    source_paths = (
        Path(__file__), Path(__file__).parent / "reconfigurable_navigation/locomotion_runtime.py",
        Path(__file__).parent / "reconfigurable_navigation/piper_locomotion_runtime.py",
        Path(__file__).parent / "robots/go2_piper/scene.xml",
        Path(__file__).parent / "robots/go2_piper/go2piper.xml",
    )
    mesh_paths = sorted(path for path in (Path(__file__).parent / "robots/go2_piper/assets").rglob("*") if path.is_file())
    report = {
        "schema_version": 1,
        "scope": "PIPER direct-PD MuJoCo locomotion smoke; not native Isaac or hardware acceptance",
        "policy": args.policy,
        "initialization": args.initialization,
        "robot_profile_sha256": sha256(args.robot_profile) if args.robot_profile is not None else None,
        "robot_profile_loader_sha256": sha256(Path(__file__).parent / "reconfigurable_navigation/piper_robot_profile.py"),
        "command": list(args.velocity),
        "ee_position_command": list(args.ee_position),
        "settle_seconds": args.settle,
        "policy_seconds": args.duration,
        "successful_episodes": sum(result["passed"] for result in results),
        "evaluated_episodes": len(results),
        "failures": dict(Counter(result["reason"] for result in results if not result["passed"])),
        "criteria": {"min_base_height": 0.16, "max_tilt_rad": float(np.pi / 3),
                 "illegal_ground_normal_force": 1.0, "max_mean_xy_error": max(0.12, 0.5 * np.linalg.norm(args.velocity[:2])),
                 "max_mean_yaw_rate_error": 0.2},
        "source_sha256": {str(path): sha256(path) for path in source_paths},
        "mesh_sha256": {str(path): sha256(path) for path in mesh_paths},
        "backend": {"mujoco": mujoco.__version__, "torch": torch.__version__, "numpy": np.__version__,
                    "cpu_capability": torch.backends.cpu.get_cpu_capability(), "torch_threads": torch.get_num_threads()},
        "results": results,
    }
    output = json.dumps(report, indent=2, allow_nan=False)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("x") as destination:
            destination.write(output + "\n")
    if args.output_json is None:
        print(output)
    else:
        print(json.dumps({"output": str(args.output_json), "successful_episodes": report["successful_episodes"],
                          "evaluated_episodes": len(results), "failures": report["failures"]}, indent=2))
    return 0 if all(result["passed"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())