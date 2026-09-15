"""Compare recorded native PIPER states with the MuJoCo asset profile."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from check_piper_startup import sha256
from reconfigurable_navigation.piper_locomotion_runtime import PiperLocomotionRuntime
from reconfigurable_navigation.piper_robot_profile import PROFILE_PATH


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-states", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        parser.error("Refusing to overwrite an alignment report")
    native = json.loads(args.native_states.read_text())
    runtime = PiperLocomotionRuntime(profile_path=PROFILE_PATH)
    model, data = runtime.model, runtime.data
    body_names = [{"base": "base_link", "link0": "Piper"}.get(name, name) for name in native["body_names"]]
    joints = [model.joint(name).id for name in native["joint_names"]]
    dofs = model.jnt_dofadr[joints]
    positions = model.jnt_qposadr[joints]
    columns = [*range(6), *dofs]
    errors = {"position": 0.0, "quaternion": 0.0, "jacobian": 0.0, "gravity": 0.0, "mass": 0.0, "inertia": 0.0, "center_of_mass": 0.0}
    for index, name in enumerate(body_names):
        errors["mass"] = max(errors["mass"], abs(model.body(name).mass[0] - native["masses"][index]))
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, model.body(name).iquat)
        rotation = rotation.reshape(3, 3)
        inertia = rotation @ np.diag(model.body(name).inertia) @ rotation.T
        errors["inertia"] = max(errors["inertia"], float(np.max(np.abs(inertia.flatten() - native["inertias"][index]))))
        errors["center_of_mass"] = max(errors["center_of_mass"], float(np.max(np.abs(model.body(name).ipos - native["centers_of_mass"][index][:3]))))
    for sample in native["samples"]:
        data.qpos[:3] = sample["root_position"]
        data.qpos[3:7] = sample["root_quaternion"]
        data.qpos[positions] = sample["joint_positions"]
        data.qvel.fill(0)
        mujoco.mj_forward(model, data)
        for index, name in enumerate(body_names):
            errors["position"] = max(errors["position"], float(np.max(np.abs(data.body(name).xpos - sample["body_positions"][index]))))
            expected = np.asarray(sample["body_quaternions"][index])
            expected /= np.linalg.norm(expected)
            actual = data.body(name).xquat
            errors["quaternion"] = max(errors["quaternion"], float(min(np.linalg.norm(actual - expected), np.linalg.norm(actual + expected))))
        linear, angular = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, linear, angular, model.body("end_effector").id)
        actual_jacobian = np.concatenate((linear, angular))[:, columns]
        offset = data.xipos[runtime.base_id] - data.xpos[runtime.base_id]
        cross_matrix = np.array([[0, -offset[2], offset[1]], [offset[2], 0, -offset[0]], [-offset[1], offset[0], 0]])
        actual_jacobian[:3, 3:6] += cross_matrix
        errors["jacobian"] = max(errors["jacobian"], float(np.max(np.abs(actual_jacobian - sample["hand_jacobian"]))))
        gravity = np.asarray(sample["gravity_forces"])
        actual_gravity = data.qfrc_bias[dofs] if gravity.shape == (18,) else data.qfrc_bias[columns]
        if gravity.shape != (18,):
            actual_gravity[3:6] -= np.cross(offset, actual_gravity[:3])
        errors["gravity"] = max(errors["gravity"], float(np.max(np.abs(actual_gravity - gravity))))
    limits = {"position": 1e-4, "quaternion": 1e-4, "jacobian": 1e-3, "gravity": 1e-3, "mass": 1e-5, "inertia": 1e-5, "center_of_mass": 1e-5}
    report = {"schema_version": 1, "samples": len(native["samples"]), "errors": errors, "tolerances": limits,
              "floating_base_reference": "PhysX base COM; MuJoCo Jacobian and wrench shifted from base origin",
              "passed": all(errors[name] <= limit for name, limit in limits.items()),
              "native_states_sha256": sha256(args.native_states), "profile_sha256": sha256(PROFILE_PATH), "check_sha256": sha256(Path(__file__))}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x") as destination:
        json.dump(report, destination, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())