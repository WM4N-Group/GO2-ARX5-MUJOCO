"""Compare PIPER CLIMB observations against recorded native states."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from check_piper_startup import sha256
from reconfigurable_navigation.piper_box_runtime import make_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-states", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        parser.error("Refusing to overwrite a report")
    native = json.loads(args.native_states.read_text())
    runtime = make_episode("climb", args.policy, 0)
    model, data = runtime.model, runtime.data
    joints = [model.joint(name).id for name in native["joint_names"]]
    terms = {"linear_velocity": (0, 3), "angular_velocity": (3, 6), "gravity": (6, 9),
             "command": (9, 12), "joint_positions": (12, 30), "joint_velocities": (30, 48),
             "last_action": (48, 66), "height_scan": (66, 253)}
    errors = dict.fromkeys(terms, 0.0)
    for sample in native["samples"]:
        data.qpos[:7] = [*sample["root_position"], *sample["root_quaternion"]]
        data.qpos[model.jnt_qposadr[joints]] = sample["joint_positions"]
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
        rotation = rotation.reshape(3, 3)
        angular = np.array(sample["root_angular_velocity_world"])
        data.qvel[:3] = np.array(sample["root_linear_velocity_world"]) - np.cross(angular, rotation @ model.body_ipos[runtime.base_id])
        data.qvel[3:6] = rotation.T @ angular
        data.qvel[model.jnt_dofadr[joints]] = sample["joint_velocities"]
        mujoco.mj_forward(model, data)
        delta = np.array([1.35, 0.0]) - data.qpos[:2]
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        relative = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]]) @ delta
        runtime.velocity_command[:] = [np.clip(0.8*relative[0], -0.15, 0.30), np.clip(0.8*relative[1], -0.12, 0.12), 0.0]
        if np.linalg.norm(delta) < 0.1:
            runtime.velocity_command.fill(0)
        expected = np.asarray(sample["policy_observation"])
        runtime.last_action[:] = expected[48:66]
        actual = runtime.observation()
        for name, (start, end) in terms.items():
            errors[name] = max(errors[name], float(np.max(np.abs(actual[start:end] - expected[start:end]))))
    report = {"schema_version": 1, "samples": len(native["samples"]), "max_abs_errors": errors,
              "tolerance": 1e-4, "passed": max(errors.values()) <= 1e-4,
              "native_sha256": sha256(args.native_states), "script_sha256": sha256(Path(__file__))}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x") as destination:
        json.dump(report, destination, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())