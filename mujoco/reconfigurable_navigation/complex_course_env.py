"""Privileged MuJoCo environment for the PUSH-to-CLIMB MVP course."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np

from .env import BlockedPassageEnv
from .representations import ObjectState, ObjectType, OracleObservation


XML_PATH = Path(__file__).resolve().parent / "complex_course.xml"


class ComplexCourseEnv(BlockedPassageEnv):
    """A corridor blocker followed by a right-angle raised platform."""

    def __init__(self) -> None:
        self.reset_count = 0
        super().__init__(xml_path=XML_PATH)
        self.base_collision_geom_ids = frozenset(
            geom_id
            for geom_id, body_id in enumerate(self.model.geom_bodyid)
            if int(body_id) == self.base_body_id
            and self.model.geom_contype[geom_id] != 0
        )
        self.terrain_geom_ids = frozenset(
            geom_id
            for geom_id, body_id in enumerate(self.model.geom_bodyid)
            if int(body_id) == 0 and self.model.geom_contype[geom_id] != 0
        )

    @property
    def push_target(self) -> np.ndarray:
        return np.array([1.5, 0.0, 0.0], dtype=np.float32)

    def reset(self, seed: int | None = None) -> OracleObservation:
        self.reset_count += 1
        return super().reset(seed=seed)

    def observe(self) -> OracleObservation:
        observation = super().observe()
        box = next(obj for obj in observation.objects if obj.object_id == 10)
        platform = ObjectState(
            object_id=20,
            object_type=ObjectType.PLATFORM,
            center=np.array([0.8, 4.1, 0.08], dtype=np.float32),
            size=np.array([4.0, 5.2, 0.16], dtype=np.float32),
            supportable=True,
            climb_entry_pose=np.array(
                [0.8, 1.05, np.pi / 2.0], dtype=np.float32
            ),
            climb_landing_pose=np.array(
                [0.8, 4.70, np.pi / 2.0], dtype=np.float32
            ),
        )
        walls = (
            ObjectState(
                1,
                ObjectType.STATIC_OBSTACLE,
                np.array([-1.35, 0.9, 0.3]),
                np.array([3.3, 0.2, 0.6]),
            ),
            ObjectState(
                2,
                ObjectType.STATIC_OBSTACLE,
                np.array([-0.5, -0.9, 0.3]),
                np.array([5.0, 0.2, 0.6]),
            ),
            ObjectState(
                3,
                ObjectType.STATIC_OBSTACLE,
                np.array([-3.0, 0.0, 0.3]),
                np.array([0.2, 2.0, 0.6]),
            ),
            ObjectState(
                4,
                ObjectType.STATIC_OBSTACLE,
                np.array([2.2, 0.0, 0.3]),
                np.array([0.2, 2.0, 0.6]),
            ),
        )
        base_contact = any(
            {int(contact.geom1), int(contact.geom2)}
            & self.base_collision_geom_ids
            and {int(contact.geom1), int(contact.geom2)}
            & self.terrain_geom_ids
            for contact in self.data.contact[: self.data.ncon]
        )
        return replace(
            observation,
            goal=np.array([0.8, 6.0, 0.0, 1.0], dtype=np.float32),
            objects=(*walls, box, platform),
            state_valid=observation.state_valid and not base_contact,
            illegal_collision=observation.illegal_collision or base_contact,
        )