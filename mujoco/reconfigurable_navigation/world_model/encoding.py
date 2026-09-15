"""Versioned start-only inputs and masked terminal supervision."""

import numpy as np


SCHEMA = "box_support_tabular_v1"
ROBOT_FIELDS = ("z", "sin_roll", "cos_roll", "sin_pitch", "cos_pitch", "sin_yaw", "cos_yaw", "vx", "vy", "vz", "yaw_rate", "supported_feet", "valid")
OBJECT_FIELDS = ("dx", "dy", "dz", "size_x", "size_y", "size_z", "sin_yaw", "cos_yaw", "mass", "vx", "vy", "vz", "wx", "wy", "wz", "movable", "supportable", "confidence", "contact", "ee_contact", "body_contact")
CAPABILITY_FIELDS = ("body_length", "body_width", "max_step_height", "max_gap_width", "max_slope", "max_push_force", "max_pushable_mass", "max_linear_speed")
OPERATIONS = (("PUSH", 10), ("NAV", 10), ("CLIMB", 10), ("NAV", 20), ("CLIMB", 20))
FEATURE_NAMES = (
    tuple(f"robot_{name}" for name in ROBOT_FIELDS)
    + ("goal_dx", "goal_dy", "goal_dz", "goal_valid")
    + tuple(f"{role}_{name}" for role in ("box", "platform") for name in OBJECT_FIELDS)
    + tuple(f"capability_{name}" for name in CAPABILITY_FIELDS)
    + ("target_dx", "target_dy", "target_dz", "target_has_z", "target_sin_yaw", "target_cos_yaw", "skill_nav", "skill_push", "skill_climb", "anchor_box", "anchor_platform")
    + ("previous_none", "previous_nav", "previous_push", "previous_climb")
    + tuple(f"completed_{kind.lower()}_{object_id}" for kind, object_id in OPERATIONS)
    + ("box_friction", "illegal_collision_before", "suffix_enabled", "suffix_control_budget", "suffix_skill_budget")
)
REGRESSION_NAMES = ("robot_dx", "robot_dy", "robot_dz", "robot_sin_dyaw", "robot_cos_dyaw", "box_dx", "box_dy", "box_dz", "box_sin_dyaw", "box_cos_dyaw", "log_skill_seconds", "log_total_seconds")
BINARY_NAMES = ("skill_success", "illegal_collision", "oracle_task_success")


def _vector(value, size, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} requires {size} finite values")
    return result


def _objects(observation):
    result = {int(obj["object_id"]): obj for obj in observation["objects"]}
    if set(result) != {10, 20} or len(observation["objects"]) != 2:
        raise ValueError("This encoder requires exactly one box and one platform")
    return result


def encode_inputs(observation, action, *, previous_skill=None, completed_operations=(), scene_parameters=None, suffix_budget=None):
    robot = _vector(observation["robot_state"], 12, "robot_state")
    goal = _vector(observation["goal"], 4, "goal")
    skill = int(action["skill"])
    if skill not in (0, 1, 2):
        raise ValueError("World model supports executable NAV/PUSH/CLIMB requests only")
    target = _vector(action["target_pose"], 4 if skill == 2 else 3, "target_pose")
    anchor = int(action["object_id"] if skill == 1 else action["support_id"])
    if anchor not in (10, 20):
        raise ValueError("Unsupported target object")
    values = [robot[2], np.sin(robot[3]), np.cos(robot[3]), np.sin(robot[4]), np.cos(robot[4]), np.sin(robot[5]), np.cos(robot[5]), *robot[6:12], *(goal[:3] - robot[:3]), goal[3]]
    objects = _objects(observation)
    for object_id in (10, 20):
        obj = objects[object_id]
        center = _vector(obj["center"], 3, "object center")
        size = _vector(obj["size"], 3, "object size")
        linear = _vector(obj["linear_velocity"], 3, "object velocity")
        angular = _vector(obj["angular_velocity"], 3, "object angular velocity")
        values.extend([*(center - robot[:3]), *size, np.sin(obj["yaw"]), np.cos(obj["yaw"]), obj["mass"], *linear, *angular, obj["movable"], obj["supportable"], obj["confidence"], object_id in observation["contact_object_ids"], object_id in observation["end_effector_contact_object_ids"], object_id in observation["body_contact_object_ids"]])
    values.extend(observation["capability"][name] for name in CAPABILITY_FIELDS)
    values.extend([*(target[:2] - robot[:2]), target[2] - robot[2] if skill == 2 else 0.0, skill == 2, np.sin(target[-1] - robot[5]), np.cos(target[-1] - robot[5]), *(skill == index for index in range(3)), anchor == 10, anchor == 20])
    previous = 0 if previous_skill is None else int(previous_skill) + 1
    if previous not in range(4):
        raise ValueError("Unsupported previous skill")
    values.extend(previous == index for index in range(4))
    completed = {(str(getattr(kind, "name", kind)), int(object_id)) for kind, object_id in completed_operations}
    values.extend(operation in completed for operation in OPERATIONS)
    if scene_parameters is None or "box_friction" not in scene_parameters:
        raise ValueError("Privileged box friction is required")
    values.extend([scene_parameters["box_friction"], observation["illegal_collision"], suffix_budget is not None, 0 if suffix_budget is None else suffix_budget["max_control_steps"], 0 if suffix_budget is None else suffix_budget["max_skills"]])
    return _vector(values, len(FEATURE_NAMES), "features").astype(np.float32)


def encode_example(candidate, source):
    if not candidate["executed"] or candidate["transition"] is None:
        return None
    transition = candidate["transition"]
    before, after = transition["observation_before"], transition["observation_after"]
    metadata = source["metadata"]
    continuation = candidate.get("continuation") or {}
    features = encode_inputs(before, candidate["action"], previous_skill=transition["previous_skill"], completed_operations=metadata.get("start_context", {}).get("completed_operations", ()), scene_parameters=metadata["scene_parameters"], suffix_budget=continuation.get("budget"))
    robot_before = _vector(before["robot_state"], 12, "robot before")
    robot_after = _vector(after["robot_state"], 12, "robot after")
    box_before, box_after = _objects(before)[10], _objects(after)[10]
    robot_yaw_delta = robot_after[5] - robot_before[5]
    box_yaw_delta = box_after["yaw"] - box_before["yaw"]
    duration = float(transition["elapsed_sim_time"])
    if not np.isfinite(duration) or duration < 0.0:
        raise ValueError("Invalid skill duration")
    regression = np.array([*(robot_after[:3] - robot_before[:3]), np.sin(robot_yaw_delta), np.cos(robot_yaw_delta), *(_vector(box_after["center"], 3, "box after") - _vector(box_before["center"], 3, "box before")), np.sin(box_yaw_delta), np.cos(box_yaw_delta), np.log1p(duration), 0.0], dtype=np.float32)
    complete = not transition["interrupted"] and not candidate.get("truncated", False)
    regression_mask = np.array([complete and candidate["label_validity"]["dynamics"]] * 10 + [complete, False], dtype=bool)
    total_time = continuation.get("total_sim_time")
    if continuation.get("label_validity", {}).get("total_cost", False) and total_time is not None:
        if not np.isfinite(total_time) or total_time < 0.0:
            raise ValueError("Invalid total duration")
        regression[-1] = np.log1p(total_time)
        regression_mask[-1] = True
    labels = [candidate["skill_success"], candidate["process_labels"].get("illegal_collision"), continuation.get("oracle_task_success")]
    binary = np.array([0.0 if value is None else float(value) for value in labels], dtype=np.float32)
    binary_mask = np.array([complete and candidate["label_validity"]["skill_success"] and labels[0] is not None, candidate["label_validity"].get("illegal_collision", False) and labels[1] is not None, continuation.get("label_validity", {}).get("oracle_task_success", False) and labels[2] is not None], dtype=bool)
    return features, regression, regression_mask, binary, binary_mask