"""Apply a provenance-bound PIPER profile to an isolated MuJoCo specification."""

import json
from pathlib import Path

import numpy as np
import mujoco


PROFILE_PATH = Path(__file__).resolve().parents[1] / "deploy/piper_robot_profile.json"
COLLISION_PROFILE_PATH = PROFILE_PATH.with_name("piper_collision_profile.json")


def apply_piper_collision_profile(spec, profile_path=COLLISION_PROFILE_PATH):
    profile = json.loads(Path(profile_path).read_text())
    if profile["schema_version"] != 1 or profile["robot"] != "go2_piper":
        raise ValueError("Expected a PIPER collision profile")
    for geom in list(spec.geoms):
        if geom.contype or geom.conaffinity:
            spec.delete(geom)
    types = {"Cube": mujoco.mjtGeom.mjGEOM_BOX, "Sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
             "Cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER, "Mesh": mujoco.mjtGeom.mjGEOM_MESH}
    for index, shape in enumerate(profile["shapes"]):
        body = spec.body(shape["body"])
        options = {"name": shape["name"], "type": types[shape["kind"]], "pos": shape["position"],
                   "quat": shape["quaternion"], "contype": 1, "conaffinity": 1, "group": 3,
                   "density": 0.0, "friction": [0.6, 0.0, 0.0], "condim": 3, "margin": 0.0}
        if shape["kind"] == "Mesh":
            name = f"piper_native_hull_{index}"
            spec.add_mesh(name=name, uservert=np.asarray(shape["vertices"]).ravel().tolist())
            options["meshname"] = name
        else:
            options["size"] = [*shape["size"], *([0.0] * (3-len(shape["size"])))]
        body.add_geom(**options)
    return profile


def apply_piper_robot_profile(spec, profile_path=PROFILE_PATH):
    profile = json.loads(Path(profile_path).read_text())
    if profile["schema_version"] != 1 or profile["robot"] != "go2_piper":
        raise ValueError("Expected a PIPER robot profile")
    for entry in profile["bodies"]:
        body = spec.body(entry["name"])
        if body is None:
            raise ValueError(f"Missing profile body: {entry['name']}")
        if "position" in entry:
            body.pos = entry["position"]
            body.quat = entry["quaternion"]
        body.explicitinertial = True
        body.mass = entry["mass"]
        body.ipos = entry["center_of_mass"]
        body.inertia = np.zeros(3)
        tensor = np.asarray(entry["inertia"]).reshape(3, 3)
        body.fullinertia = tensor[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]
    for entry in profile["joints"]:
        joint = spec.joint(entry["name"])
        joint.axis = entry["axis"]
        joint.range = entry["range"]
        joint.damping = [0.0, 0.0, 0.0]
        joint.armature = 0.01
        joint.frictionloss = 0.01
    for name in ("link7", "link8"):
        body = spec.body(name)
        body.explicitinertial = True
        body.mass = 0.0
        body.inertia = np.zeros(3)
    return profile