"""Privileged yaw-box BEV and masked object tokens for support scenes."""

import numpy as np

from .encoding import _vector


SPATIAL_SCHEMA = "box_support_objects_bev_v2"
BEV_SIZE = 128
BEV_RESOLUTION = 0.05
MAX_OBJECTS = 32
PROPRIO_FIELDS = ("joint_position", "joint_velocity", "last_action", "actuator_target")
PROPRIO_SIZE = 18 * len(PROPRIO_FIELDS) + 1
BEV_CHANNELS = ("occupied", "height_m", "height_edge_proxy", "supportable", "movable", "static_obstacle", "known", "robot_goal_hint")
CATEGORIES = ("floor", "movable_box", "static_obstacle", "platform", "bridge", "goal_marker", "other")
OBJECT_TOKEN_NAMES = (
    ("dx", "dy", "dz", "size_x", "size_y", "size_z", "movable", "supportable")
    + tuple(f"type_{name}" for name in CATEGORIES)
    + ("confidence", "sin_yaw", "cos_yaw", "vx", "vy", "vz", "wx", "wy", "wz", "mass", "friction", "contact", "ee_contact", "body_contact", "object_pointer", "support_pointer", "friction_known")
)


def _footprint(grid_x, grid_y, center, size, yaw):
    relative_x, relative_y = grid_x - center[0], grid_y - center[1]
    local_x = np.cos(yaw) * relative_x + np.sin(yaw) * relative_y
    local_y = -np.sin(yaw) * relative_x + np.cos(yaw) * relative_y
    return (np.abs(local_x) <= size[0] / 2.0) & (np.abs(local_y) <= size[1] / 2.0)


def encode_proprio(context):
    if context.get("proprio_schema") is None:
        return np.zeros(PROPRIO_SIZE, dtype=np.float32)
    if context["proprio_schema"] != "box_runtime_joint_state_v1":
        raise ValueError("Unsupported proprioceptive context schema")
    return np.concatenate([*(_vector(context[name], 18, name) for name in PROPRIO_FIELDS), [1.0]]).astype(np.float32)


def encode_scene(observation, action, scene_parameters):
    robot = _vector(observation["robot_state"], 12, "robot state")
    goal = _vector(observation["goal"], 4, "goal")
    objects = observation["objects"]
    if not 0 < len(objects) <= MAX_OBJECTS:
        raise ValueError("Object count must lie between 1 and 32")
    identities = [int(obj["object_id"]) for obj in objects]
    if len(set(identities)) != len(identities):
        raise ValueError("Object identities must be unique")
    for name in ("object_id", "support_id"):
        if int(action[name]) >= 0 and int(action[name]) not in identities:
            raise ValueError("Action pointer does not reference an observed object")
    tokens = np.zeros((MAX_OBJECTS, len(OBJECT_TOKEN_NAMES)), dtype=np.float32)
    mask = np.zeros(MAX_OBJECTS, dtype=bool)
    object_ids = np.full(MAX_OBJECTS, -1, dtype=np.int64)
    geometry = []
    for index, obj in enumerate(objects):
        center = _vector(obj["center"], 3, "object center") - robot[:3]
        size = _vector(obj["size"], 3, "object size")
        if np.any(size <= 0.0) or not np.isfinite(obj["yaw"]):
            raise ValueError("Object dimensions and yaw must be valid")
        category = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 6}.get(int(obj["object_type"]), 6)
        object_id = int(obj["object_id"])
        known_friction = object_id == 10 or category in (0, 3)
        friction = scene_parameters["box_friction"] if object_id == 10 else 0.6 if known_friction else 0.0
        values = [*center, *size, obj["movable"], obj["supportable"], *(category == item for item in range(len(CATEGORIES))), obj["confidence"], np.sin(obj["yaw"]), np.cos(obj["yaw"]), *_vector(obj["linear_velocity"], 3, "object velocity"), *_vector(obj["angular_velocity"], 3, "object angular velocity"), obj["mass"], friction, object_id in observation["contact_object_ids"], object_id in observation["end_effector_contact_object_ids"], object_id in observation["body_contact_object_ids"], object_id == action["object_id"], object_id == action["support_id"], known_friction]
        tokens[index] = _vector(values, len(OBJECT_TOKEN_NAMES), "object token")
        mask[index], object_ids[index] = True, object_id
        geometry.append((float(obj["center"][2] + size[2] / 2.0), object_id, center, size, float(obj["yaw"]), bool(obj["supportable"]), bool(obj["movable"]), category))
    coordinates = (np.arange(BEV_SIZE, dtype=np.float64) + 0.5 - BEV_SIZE / 2.0) * BEV_RESOLUTION
    grid_x, grid_y = np.meshgrid(coordinates, coordinates, indexing="xy")
    bev = np.zeros((len(BEV_CHANNELS), BEV_SIZE, BEV_SIZE), dtype=np.float32)
    bev[3] = 1.0
    bev[6] = 1.0
    for top, _object_id, center, size, yaw, supportable, movable, category in sorted(geometry, key=lambda item: (item[0], item[1])):
        footprint = _footprint(grid_x, grid_y, center, size, yaw)
        visible = footprint & (top >= bev[1])
        bev[1, visible] = max(top, 0.0)
        bev[3, visible] = supportable
        bev[4, visible] = movable
        bev[5, visible] = category in (2, 3)
    bev[0] = bev[1] > 0.0
    gradient_y, gradient_x = np.gradient(bev[1], BEV_RESOLUTION)
    bev[2] = np.clip(np.hypot(gradient_x, gradient_y), 0.0, 1.0)
    body = (observation["capability"]["body_length"], observation["capability"]["body_width"])
    bev[7, _footprint(grid_x, grid_y, (0.0, 0.0), body, robot[5])] = 1.0
    if goal[3]:
        delta = goal[:2] - robot[:2]
        bev[7, (grid_x - delta[0]) ** 2 + (grid_y - delta[1]) ** 2 <= 0.1 ** 2] = -1.0
    return {"bev": bev, "objects": tokens, "object_mask": mask, "object_ids": object_ids}