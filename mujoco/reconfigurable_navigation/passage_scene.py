"""Parameterized physical variants of the original blocked passage."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from .env import BlockedPassageEnv
from .representations import Capability, ObjectState, ObjectType, OracleObservation


@dataclass(frozen=True)
class PassageScene:
    corridor_width: float = 1.6
    box_size: tuple[float, float, float] = (0.44, 1.44, 0.60)
    box_mass: float = 5.0
    box_friction: float = 0.005
    robot_pose: tuple[float, float, float] = (-2.30, 0.0, 0.0)
    box_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)
    goal_pose: tuple[float, float, float] = (1.8, 0.0, 0.0)
    push_target_pose: tuple[float, float, float] = (2.45, 0.0, 0.0)

    def __post_init__(self) -> None:
        for name in ("box_size", "robot_pose", "box_pose", "goal_pose", "push_target_pose"):
            values = np.asarray(getattr(self, name), dtype=np.float64)
            if values.shape != (3,) or not np.isfinite(values).all():
                raise ValueError(f"{name} must contain three finite values")
            object.__setattr__(self, name, tuple(float(value) for value in values))
        for name in ("corridor_width", "box_mass", "box_friction"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        if min(self.box_size) <= 0:
            raise ValueError("box_size must be positive")
        yaw = self.box_pose[2]
        rotation = np.abs([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        half_extent = rotation @ (np.asarray(self.box_size[:2]) / 2.0)
        if np.any(np.abs(self.box_pose[:2]) + half_extent >= [2.9, self.corridor_width / 2]):
            raise ValueError("Initial box intersects corridor walls")
        for name in ("robot_pose", "goal_pose", "push_target_pose"):
            position = np.asarray(getattr(self, name)[:2])
            if np.any(np.abs(position) >= [2.9, self.corridor_width / 2]):
                raise ValueError(f"{name} must lie inside the corridor")

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def scene_id(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode()).hexdigest()[:16]


def passage_sweep(seed: int) -> dict[str, PassageScene]:
    generator = np.random.default_rng(seed)
    baseline = PassageScene(
        robot_pose=(-2.30, generator.uniform(-0.12, 0.12), generator.uniform(-0.08, 0.08)),
        box_pose=(generator.uniform(-0.12, 0.12), generator.uniform(-0.02, 0.02), 0.0),
    )
    return {
        "baseline": baseline,
        "mass_low": replace(baseline, box_mass=3.0),
        "mass_over_limit": replace(baseline, box_mass=10.0),
        "friction_high": replace(baseline, box_friction=0.15),
        "corridor_wide": replace(baseline, corridor_width=1.8),
        "box_narrow": replace(baseline, box_size=(0.44, 1.2, 0.60)),
        "box_yaw": replace(baseline, box_pose=(*baseline.box_pose[:2], 0.04)),
        "box_position": replace(baseline, box_pose=(0.15, baseline.box_pose[1], 0.0)),
        "robot_start": replace(baseline, robot_pose=(-2.15, *baseline.robot_pose[1:])),
        "goal_position": replace(baseline, goal_pose=(1.6, 0.0, 0.0)),
    }


class ParameterizedPassageEnv(BlockedPassageEnv):
    def __init__(self, scene: PassageScene | None = None, capability: Capability | None = None) -> None:
        self.scene = scene or PassageScene()
        self.reset_count = 0
        super().__init__(capability=capability)

    def _load_model(self, xml_path: Path | str) -> mujoco.MjModel:
        spec = mujoco.MjSpec.from_file(str(xml_path))
        half_width = self.scene.corridor_width / 2.0
        spec.body("wall_left").pos = [0.0, half_width + 0.1, 0.3]
        spec.body("wall_right").pos = [0.0, -half_width - 0.1, 0.3]
        for name in ("wall_start_geom", "wall_goal_geom"):
            spec.geom(name).size = [0.1, half_width + 0.2, 0.3]
        box = spec.geom("movable_box_geom")
        box.size = np.asarray(self.scene.box_size) / 2.0
        box.mass = self.scene.box_mass
        box.friction = [self.scene.box_friction, 0.001, 0.0001]
        spec.body("movable_box").pos = [0.0, 0.0, self.scene.box_size[2] / 2.0]
        spec.body("goal_marker").pos = [*self.scene.goal_pose[:2], 0.012]
        return spec.compile()

    def reset(self, seed: int | None = None) -> OracleObservation:
        super().reset(seed)
        self._set_free_joint(
            self.robot_qpos_adr, np.array([*self.scene.robot_pose[:2], 0.445]), self.scene.robot_pose[2],
        )
        self._set_free_joint(
            self.box_qpos_adr, np.array([*self.scene.box_pose[:2], self.scene.box_size[2] / 2.0]), self.scene.box_pose[2],
        )
        mujoco.mj_forward(self.model, self.data)
        self.reset_count += 1
        observation = self.observe()
        if observation.illegal_collision or observation.contact_object_ids:
            raise ValueError("Initial robot collides with a wall or the box")
        return observation

    @property
    def push_target(self) -> np.ndarray:
        return np.asarray(self.scene.push_target_pose, dtype=np.float32)

    def observe(self) -> OracleObservation:
        observation = super().observe()
        box = next(obj for obj in observation.objects if obj.object_id == 10)
        walls = []
        for object_id, name in enumerate(("wall_left_geom", "wall_right_geom", "wall_start_geom", "wall_goal_geom"), 1):
            geom_id = self._geom_id(name)
            walls.append(ObjectState(
                object_id, ObjectType.STATIC_OBSTACLE,
                self.data.geom_xpos[geom_id].copy(), self.model.geom_size[geom_id].copy() * 2.0,
            ))
        return replace(
            observation,
            goal=np.array([*self.scene.goal_pose, 1.0], dtype=np.float32),
            objects=(*walls, replace(
                box, size=self.model.geom_size[self.box_geom_id].copy() * 2.0,
                mass=float(self.model.body_mass[self.box_body_id]),
            )),
        )