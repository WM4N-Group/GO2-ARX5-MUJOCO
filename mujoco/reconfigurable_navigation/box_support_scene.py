"""Parameterized single-box support layouts and deterministic family grouping."""

from dataclasses import asdict, dataclass, replace
import hashlib
import json

import numpy as np


BOX_SUPPORT_FAMILIES = frozenset(("box_support_nominal", "box_support_aligned", "box_support_left_offset", "box_support_right_offset"))


@dataclass(frozen=True)
class BoxSupportScene:
    robot_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)
    box_pose: tuple[float, float, float] = (1.10, 0.0, 0.0)
    box_size: tuple[float, float, float] = (1.2, 1.2, 0.20)
    box_mass: float = 5.0
    box_friction: float = 0.4
    platform_pose: tuple[float, float, float] = (3.3, 0.0, 0.0)
    platform_size: tuple[float, float, float] = (2.0, 1.6, 0.40)

    def __post_init__(self):
        for name in ("robot_pose", "box_pose", "box_size", "platform_pose", "platform_size"):
            values = np.asarray(getattr(self, name), dtype=np.float64)
            if values.shape != (3,) or not np.isfinite(values).all():
                raise ValueError(f"{name} requires three finite values")
            object.__setattr__(self, name, tuple(float(value) for value in values))
        for name in ("box_mass", "box_friction"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if min((*self.box_size, *self.platform_size)) <= 0.0 or self.platform_size[2] <= self.box_size[2]:
            raise ValueError("Positive dimensions and a platform above the box are required")
        if abs(self.platform_pose[2]) > 1e-8:
            raise ValueError("Frozen PUSH supports positive-X platform approaches only")
        yaw = self.box_pose[2]
        rotation = np.abs([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        extent = rotation @ (np.asarray(self.box_size[:2]) / 2.0)
        if np.all(np.abs(np.asarray(self.box_pose[:2]) - self.platform_pose[:2]) < extent + np.asarray(self.platform_size[:2]) / 2.0):
            raise ValueError("Initial box and platform overlap")

    def to_dict(self):
        return asdict(self)

    @property
    def scene_id(self):
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode()).hexdigest()[:16]

    @property
    def scene_family(self):
        lateral = self.platform_pose[1] - self.box_pose[1]
        return "box_support_aligned" if abs(lateral) < 1e-8 else "box_support_left_offset" if lateral > 0.0 else "box_support_right_offset"

    @property
    def dataset_split(self):
        return {"box_support_aligned": "train", "box_support_left_offset": "validation", "box_support_right_offset": "test"}[self.scene_family]


def box_support_suite():
    baseline = BoxSupportScene()
    return {
        "aligned": baseline,
        "aligned_far": replace(baseline, platform_pose=(3.38, 0.0, 0.0)),
        "left": replace(baseline, platform_pose=(3.3, 0.04, 0.0)),
        "left_far_narrow": replace(baseline, platform_pose=(3.36, 0.06, 0.0), platform_size=(2.0, 1.5, 0.4)),
        "right": replace(baseline, platform_pose=(3.3, -0.04, 0.0)),
        "right_near_wide": replace(baseline, platform_pose=(3.24, -0.06, 0.0), platform_size=(2.0, 1.7, 0.4)),
    }