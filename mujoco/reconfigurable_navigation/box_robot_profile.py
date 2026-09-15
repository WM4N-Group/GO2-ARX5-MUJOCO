"""Apply the measured Isaac box-task inertial profile to an isolated MjSpec."""

import json
import hashlib
import os
from pathlib import Path

import numpy as np


PROFILE_PATH = Path(__file__).resolve().parents[1] / "deploy/box_robot_profile.json"


def box_runtime_provenance():
    import torch

    root = PROFILE_PATH.parents[1]
    paths = [PROFILE_PATH, root / "robots/go2_arx5/go2_arx5.xml", root / "run_box_support_sequence.py", root / "run_box_support_executor.py", root / "check_box_support_executor.py"]
    paths.extend(root / "reconfigurable_navigation" / name for name in (
        "box_robot_profile.py", "box_push_runtime.py", "box_climb_runtime.py",
        "box_navigation_runtime.py", "climb_runtime.py", "locomotion_runtime.py",
        "box_support_control.py",
        "box_support_env.py",
        "box_support_geometry.py",
        "box_support_scene.py",
        "box_support_planner.py", "runtime/box_support_backend.py", "runtime/skill_backend.py", "runtime/executor.py",
        "data/transition.py", "data/events.py", "data/snapshot.py", "data/snapshot_io.py", "data/suffix.py",
    ))
    paths.append(root.parent / "source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/mdp/box_push_control.py")
    return {
        "files_sha256": {str(path.relative_to(root.parent)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "robot_mass": json.loads(PROFILE_PATH.read_text())["robot_mass"],
        "contact_force_semantics": "normal_force_world",
        "delay_initialization": "repeat_first_target",
        "numeric_backend": {
            "torch_version": torch.__version__, "numpy_version": np.__version__,
            "torch_cpu_capability": torch._C._get_cpu_capability(),
            "torch_threads": torch.get_num_threads(),
            "kernel_settings": {name: os.environ.get(name) for name in ("ATEN_CPU_CAPABILITY", "MKL_CBWR", "DNNL_MAX_CPU_ISA")},
        },
    }


def apply_box_robot_profile(spec, profile=None):
    if profile is None:
        profile = json.loads(PROFILE_PATH.read_text())
    if profile["schema_version"] != 1:
        raise ValueError("Unsupported box robot profile")
    for entry in profile["bodies"]:
        body = spec.body(entry["name"])
        if body is None:
            body = spec.body(entry["parent"]).add_body(name=entry["name"])
        if "position" in entry:
            body.pos = entry["position"]
            body.quat = entry["quaternion"]
        body.explicitinertial = True
        body.mass = entry["mass"]
        body.ipos = entry["center_of_mass"]
        body.inertia = [0.0, 0.0, 0.0]
        inertia = np.asarray(entry["inertia"]).reshape(3, 3)
        body.fullinertia = inertia[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]
    for name in ("x5_link7", "x5_link8"):
        body = spec.body(name)
        body.explicitinertial = True
        body.mass = 0.0
        body.inertia = [0.0, 0.0, 0.0]