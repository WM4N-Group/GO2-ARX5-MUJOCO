"""Validated standing-state distributions for separate CLIMB training episodes."""

import json
from pathlib import Path

import torch


def load_start_states(path, phase, device="cpu"):
    report = json.loads(Path(path).read_text())
    states = [stage["prepared_state"] for result in report["results"] for stage in result["stages"] if stage.get("succeeded") and stage.get("prepared_state", {}).get("phase") == phase]
    if not states:
        raise ValueError(f"No successful prepared states for phase {phase}")
    names = states[0]["joint_names"]
    if len(names) != 18 or len(set(names)) != 18 or any(state["joint_names"] != names for state in states):
        raise ValueError("Prepared states must share 18 unique joint names")
    fields = {}
    for name, size in (("base_pose", 7), ("base_velocity_world", 6), ("joint_position", 18), ("joint_velocity", 18)):
        values = torch.tensor([state[name] for state in states], dtype=torch.float32, device=device)
        if values.shape != (len(states), size) or not torch.isfinite(values).all():
            raise ValueError(f"Invalid prepared-state field: {name}")
        fields[name] = values
    quaternion_norm = torch.linalg.vector_norm(fields["base_pose"][:, 3:7], dim=1)
    if not torch.allclose(quaternion_norm, torch.ones_like(quaternion_norm), rtol=1e-5, atol=1e-5):
        raise ValueError("Prepared-state quaternions must be normalized")
    if (fields["base_pose"][:, 2] < 0.18).any():
        raise ValueError("Prepared starts must have standing clearance")
    return names, fields


def reset_prepared_start(env, env_ids):
    path = env.cfg.prepared_start_path
    if not path:
        return
    robot = env.scene["robot"]
    if not hasattr(env, "_climb_prepared_starts"):
        names, fields = load_start_states(path, env.cfg.prepared_start_phase, env.device)
        joint_ids, matched = robot.find_joints(names, preserve_order=True)
        if matched != names:
            raise ValueError("Prepared-start joints do not match the robot")
        env._climb_prepared_starts = joint_ids, fields
    joint_ids, fields = env._climb_prepared_starts
    indices = torch.randint(len(fields["base_pose"]), (len(env_ids),), device=env.device)
    pose = fields["base_pose"][indices].clone()
    pose[:, :3] += env.scene.env_origins[env_ids]
    positions = robot.data.default_joint_pos[env_ids].clone()
    velocities = torch.zeros_like(positions)
    positions[:, joint_ids] = fields["joint_position"][indices]
    velocities[:, joint_ids] = fields["joint_velocity"][indices]
    robot.write_root_pose_to_sim(pose, env_ids)
    robot.write_root_velocity_to_sim(fields["base_velocity_world"][indices], env_ids)
    robot.write_joint_state_to_sim(positions, velocities, env_ids=env_ids)