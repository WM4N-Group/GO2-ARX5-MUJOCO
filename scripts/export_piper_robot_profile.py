"""Export PIPER USD mass and joint frames using usd-core==25.11 on CPU."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
USD = ROOT / "source/LeggedManip_Lab/LeggedManip_Lab/assets/go2_piper/go2_piper.usd"


def rotation(quaternion):
    values = np.array([*quaternion.GetImaginary(), quaternion.GetReal()], dtype=float)
    return Rotation.from_quat(values).as_matrix() if np.linalg.norm(values) else np.eye(3)


def frame(position, quaternion):
    transform = np.eye(4)
    transform[:3, :3] = rotation(quaternion)
    transform[:3, 3] = position
    return transform


def mapped(name):
    return {"base": "base_link", "link0": "Piper"}.get(name, name)


def export_profile(startup_report):
    stage = Usd.Stage.Open(str(USD))
    if UsdGeom.GetStageMetersPerUnit(stage) != 1.0 or UsdGeom.GetStageUpAxis(stage) != "Z":
        raise ValueError("Expected Z-up PIPER USD in meters")
    measured = dict(zip(startup_report["body_names"], startup_report["body_masses"]))
    joint_frames = {}
    joints = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        parent = joint.GetBody0Rel().GetTargets()
        child = joint.GetBody1Rel().GetTargets()
        if len(parent) != 1 or len(child) != 1:
            raise ValueError(f"Unsupported joint: {prim.GetPath()}")
        local = frame(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get()) @ np.linalg.inv(
            frame(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get()),
        )
        quaternion = Rotation.from_matrix(local[:3, :3]).as_quat(scalar_first=True)
        joint_frames[child[0].name] = {"parent": mapped(parent[0].name), "position": local[:3, 3].tolist(), "quaternion": quaternion.tolist()}
        if prim.IsA(UsdPhysics.RevoluteJoint):
            revolute = UsdPhysics.RevoluteJoint(prim)
            axis = np.eye(3)[("X", "Y", "Z").index(revolute.GetAxisAttr().Get())]
            axis = rotation(joint.GetLocalRot1Attr().Get()) @ axis
            joints.append({"name": prim.GetName(), "axis": axis.tolist(), "range": np.deg2rad([revolute.GetLowerLimitAttr().Get(), revolute.GetUpperLimitAttr().Get()]).tolist()})
    bodies = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.MassAPI):
            continue
        api = UsdPhysics.MassAPI(prim)
        name = prim.GetName()
        mass = float(api.GetMassAttr().Get())
        inferred = mass <= 0.0
        if inferred:
            if name != "end_effector" or measured.get(name) != 1.0:
                raise ValueError(f"Unverified default mass: {name}")
            mass = measured[name]
        if name not in measured or abs(mass - measured[name]) > 1e-5:
            raise ValueError(f"USD and unrandomized startup masses differ: {name}")
        center = np.array(api.GetCenterOfMassAttr().Get(), dtype=float)
        if not np.isfinite(center).all():
            if not inferred:
                raise ValueError(f"Unspecified center of mass: {name}")
            center = np.zeros(3)
        axes = rotation(api.GetPrincipalAxesAttr().Get())
        inertia = axes @ np.diag(api.GetDiagonalInertiaAttr().Get()) @ axes.T
        body = {"name": mapped(name), "mass": mass, "center_of_mass": center.tolist(), "inertia": inertia.flatten().tolist()}
        body.update(joint_frames.get(name, {}))
        if inferred:
            body["default_mass_source"] = "unrandomized PhysX startup report; default COM/orientation require runtime comparison"
        bodies.append(body)
    if len(bodies) != 25 or len(joints) != 18:
        raise ValueError("Unexpected PIPER body/joint count")
    return {
        "schema_version": 1, "robot": "go2_piper", "source": "USD plus unrandomized native PUSH startup masses",
        "usd_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(USD.parent.rglob("*.usd"))},
        "robot_mass": sum(body["mass"] for body in bodies), "bodies": bodies, "joints": joints,
        "hardware_calibrated": False, "runtime_inertia_comparison_passed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--startup-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.startup_report.read_text())
    if report["task"] != "GO2-PIPER-Box-Push-Hybrid-Play":
        raise ValueError("Expected the unrandomized PIPER PUSH startup report")
    profile = export_profile(report)
    profile["startup_report_sha256"] = hashlib.sha256(args.startup_report.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as destination:
        json.dump(profile, destination, indent=2, allow_nan=False)
    print(json.dumps({"robot_mass": profile["robot_mass"], "bodies": len(profile["bodies"]), "joints": len(profile["joints"]), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()