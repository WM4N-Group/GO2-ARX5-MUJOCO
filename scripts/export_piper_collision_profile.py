"""Export PIPER USD collision primitives and convex hulls on CPU."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from export_piper_robot_profile import ROOT, USD, mapped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stage = Usd.Stage.Open(str(USD))
    cache = UsdGeom.XformCache()
    shapes = []
    for body in stage.Traverse():
        if not body.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        collision_root = body.GetChild("collisions")
        if not collision_root:
            continue
        for prim in Usd.PrimRange(collision_root, Usd.TraverseInstanceProxies()):
            kind = prim.GetTypeName()
            if kind not in ("Cube", "Sphere", "Cylinder", "Mesh"):
                continue
            owner = prim
            while owner.IsValid() and not owner.HasAPI(UsdPhysics.CollisionAPI):
                owner = owner.GetParent()
            if not owner.IsValid() or not UsdPhysics.CollisionAPI(owner).GetCollisionEnabledAttr().Get():
                continue
            matrix = np.array(cache.ComputeRelativeTransform(prim, body)[0])
            shape = {"body": mapped(body.GetName()), "source": str(prim.GetPath()), "kind": kind}
            if kind == "Mesh":
                approximation = UsdPhysics.MeshCollisionAPI(owner).GetApproximationAttr().Get()
                if approximation != "convexHull":
                    raise ValueError(f"Unsupported native mesh approximation: {prim.GetPath()} {approximation}")
                vertices = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64)
                vertices = (np.c_[vertices, np.ones(len(vertices))] @ matrix)[:, :3]
                vertices = np.unique(vertices, axis=0)
                hull = ConvexHull(vertices)
                shape["vertices"] = vertices[hull.vertices].tolist()
                shape["position"] = [0.0, 0.0, 0.0]
                shape["quaternion"] = [1.0, 0.0, 0.0, 0.0]
            else:
                linear = matrix[:3, :3].T
                scales = np.linalg.norm(linear, axis=0)
                rotation = linear / scales
                if np.linalg.det(rotation) < 0:
                    rotation[:, 0] *= -1
                np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
                shape["position"] = matrix[3, :3].tolist()
                if kind == "Cube":
                    shape["size"] = (scales * UsdGeom.Cube(prim).GetSizeAttr().Get() / 2).tolist()
                elif kind == "Sphere":
                    if np.ptp(scales) > 1e-6:
                        raise ValueError("Nonuniform sphere scale")
                    shape["size"] = [float(scales[0] * UsdGeom.Sphere(prim).GetRadiusAttr().Get())]
                else:
                    cylinder = UsdGeom.Cylinder(prim)
                    axis = cylinder.GetAxisAttr().Get()
                    if axis != "Z" or abs(scales[0] - scales[1]) > 1e-6:
                        raise ValueError("Unsupported cylinder axis or scale")
                    shape["size"] = [float(scales[0] * cylinder.GetRadiusAttr().Get()), float(scales[2] * cylinder.GetHeightAttr().Get() / 2)]
                shape["quaternion"] = Rotation.from_matrix(rotation).as_quat(scalar_first=True).tolist()
            if body.GetName().endswith("_foot"):
                shape["name"] = body.GetName().removesuffix("_foot")
            else:
                shape["name"] = f"piper_collision_{len(shapes)}"
            shapes.append(shape)
    if not shapes:
        raise ValueError("No PIPER collision shapes found")
    profile = {"schema_version": 1, "robot": "go2_piper", "shapes": shapes,
               "usd_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(USD.parent.rglob("*.usd"))}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as destination:
        json.dump(profile, destination, indent=2, allow_nan=False)
    print(json.dumps({"collision_shapes": len(shapes), "mesh_hull_vertices": sum(len(shape.get("vertices", [])) for shape in shapes), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()