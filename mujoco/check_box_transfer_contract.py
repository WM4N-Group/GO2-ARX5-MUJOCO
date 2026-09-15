"""Compare MuJoCo geometry and gravity at recorded Isaac physical states."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from reconfigurable_navigation.box_robot_profile import apply_box_robot_profile


ROOT = Path(__file__).resolve().parent


def body_name(name):
    if name == "link0":
        return "x5_base_link"
    if name == "end_effector":
        return "x5_end_effector"
    return f"x5_{name}" if name.startswith("link") else name


def rotation(quaternion):
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, np.asarray(quaternion, dtype=np.float64))
    return matrix.reshape(3, 3)


def profile_from_states(states, source):
    masses = np.asarray(states["body_masses"])
    centers = np.asarray(states["body_coms_xyzw"])[..., :3]
    inertias = np.asarray(states["body_inertias"])
    for values in (masses, centers, inertias):
        np.testing.assert_allclose(values, np.broadcast_to(values[0], values.shape), atol=1e-6, rtol=1e-6)
    if not all(states["ready"]) or any(states["failed"]):
        raise ValueError("Source contact preparation did not pass")
    names = states["body_names"]
    positions = states["snapshots"][0]["body_position"][0]
    quaternions = states["snapshots"][0]["body_quaternion"][0]
    entries = []
    for index, name in enumerate(names):
        entry = {"name": body_name(name), "mass": float(masses[0, index]), "center_of_mass": centers[0, index].tolist(), "inertia": inertias[0, index].tolist()}
        parent = "base" if name == "link0" else "link6" if name == "end_effector" else name.replace("foot", "calf") if name.endswith("_foot") else None
        if parent is not None:
            parent_index = names.index(parent)
            parent_rotation = rotation(quaternions[parent_index])
            local_rotation = parent_rotation.T @ rotation(quaternions[index])
            local_quaternion = np.empty(4)
            mujoco.mju_mat2Quat(local_quaternion, local_rotation.ravel())
            entry.update(parent=body_name(parent), position=(parent_rotation.T @ (np.asarray(positions[index]) - positions[parent_index])).tolist(), quaternion=local_quaternion.tolist())
        entries.append(entry)
    return {"schema_version": 1, "source_state_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "source_task": "GO2-ARX5-Box-Push-Hybrid-Play", "robot_mass": float(masses[0].sum()), "bodies": entries}


def compare(states, profile):
    spec = mujoco.MjSpec.from_file(str(ROOT / "robots/go2_arx5/go2_arx5.xml"))
    apply_box_robot_profile(spec, profile)
    model = spec.compile()
    data = mujoco.MjData(model)
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"x5_{name}" if name.startswith("joint") else name) for name in states["joint_names"]]
    qpos_ids = model.jnt_qposadr[joints]
    arm_dofs = [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"x5_joint{index}")] for index in range(1, 7)]
    bodies = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name(name)) for name in states["body_names"]]
    hand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "x5_end_effector")
    errors = {"body_position": 0.0, "hand_jacobian": 0.0, "arm_gravity": 0.0}
    count = 0
    for snapshot in states["snapshots"]:
        for env_id, root in enumerate(snapshot["root_state"]):
            data.qpos[:7] = root[:7]
            data.qpos[qpos_ids] = snapshot["joint_pos"][env_id]
            data.qvel.fill(0.0)
            mujoco.mj_forward(model, data)
            jacobian = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jacobian, None, hand)
            actual = {"body_position": data.xpos[bodies], "hand_jacobian": jacobian[:, arm_dofs], "arm_gravity": data.qfrc_bias[arm_dofs]}
            expected = {"body_position": snapshot["body_position"][env_id], "hand_jacobian": np.asarray(snapshot["hand_jacobian"])[env_id, :3], "arm_gravity": snapshot["arm_gravity"][env_id]}
            for name in errors:
                errors[name] = max(errors[name], float(np.max(np.abs(actual[name] - expected[name]))))
            count += 1
    result = {"matched_states": count, "max_absolute_errors": errors, "robot_mass": float(model.body_mass.sum()), "passed": errors["body_position"] < 1e-5 and errors["hand_jacobian"] < 1e-5 and errors["arm_gravity"] < 1e-4}
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--isaac-states", type=Path, required=True)
    parser.add_argument("--write-profile", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    for path in (args.write_profile, args.output_json):
        if path is not None and path.exists():
            parser.error("Output paths must not already exist")
    states = json.loads(args.isaac_states.read_text())
    profile = profile_from_states(states, args.isaac_states)
    result = compare(states, profile)
    if args.output_json is not None:
        with args.output_json.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
    if result["passed"] and args.write_profile is not None:
        args.write_profile.parent.mkdir(parents=True, exist_ok=True)
        with args.write_profile.open("x", encoding="utf-8") as stream:
            json.dump(profile, stream, indent=2, allow_nan=False)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()