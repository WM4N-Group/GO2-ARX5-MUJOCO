"""Geometric request checks for frozen support skills."""

from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

from reconfigurable_navigation.box_support_control import surface_goal
from reconfigurable_navigation.box_support_geometry import push_target_supported, surface_action_offset
from reconfigurable_navigation.data.candidates import build_candidates
from reconfigurable_navigation.representations import Capability, ObjectState, ObjectType, OracleObservation, SkillAction, SkillType


class GeometryChecks(unittest.TestCase):
    def setUp(self):
        self.capability = Capability()
        self.box = ObjectState(10, ObjectType.MOVABLE_BOX, np.array([1.1, 0.0, 0.1]), np.array([1.2, 1.2, 0.2]), movable=True, supportable=True, mass=5.0, climb_entry_pose=np.array([-0.25, 0.0, 0.0]), climb_landing_pose=np.array([1.1, 0.0, 0.48, 0.0]))
        self.platform = ObjectState(20, ObjectType.PLATFORM, np.array([3.3, 0.0, 0.2]), np.array([2.0, 1.6, 0.4]), supportable=True, climb_entry_pose=np.array([1.55, 0.0, 0.0]), climb_landing_pose=np.array([3.3, 0.0, 0.68, 0.0]))

    def test_push_region_uses_platform_width_and_forward_only_control(self):
        self.assertTrue(push_target_supported(np.array([1.74, 0.04, 0.0]), self.box, self.platform))
        narrow = replace(self.platform, size=np.array([2.0, 1.3, 0.4]))
        self.assertFalse(push_target_supported(np.array([1.74, 0.04, 0.0]), self.box, narrow))
        self.assertFalse(push_target_supported(np.array([1.74, 0.04, 0.0]), replace(self.box, center=np.array([1.8, 0.0, 0.1])), self.platform))

    def test_entry_offset_uses_rotated_surface_coordinates(self):
        surface = replace(self.box, yaw=np.pi / 2.0, climb_entry_pose=np.array([1.0, -1.0, np.pi / 2.0]))
        action = SkillAction(SkillType.NAV, [1.04, -0.94, np.pi / 2.0 + 0.02], support_id=10)
        np.testing.assert_allclose(surface_action_offset(action, surface, self.capability), [0.06, -0.04, 0.02], atol=1e-6)

    def test_platform_approach_must_fit_on_current_box(self):
        action = SkillAction(SkillType.NAV, self.platform.climb_entry_pose, support_id=20)
        self.assertIsNone(surface_action_offset(action, self.platform, self.capability, self.box))
        moved = replace(self.box, center=np.array([1.65, 0.0, 0.1]))
        np.testing.assert_array_equal(surface_action_offset(action, self.platform, self.capability, moved), np.zeros(3))

    def test_climb_keeps_height_heading_and_interior_constraints(self):
        action = SkillAction(SkillType.CLIMB, [1.16, 0.04, 0.48, 0.0], support_id=10)
        np.testing.assert_allclose(surface_action_offset(action, self.box, self.capability), [0.06, 0.04], atol=1e-6)
        for offset in ([0.2, 0.0, 0.0, 0.0], [0.0, 0.0, 0.1, 0.0], [0.0, 0.0, 0.0, 0.1]):
            invalid = SkillAction(SkillType.CLIMB, self.box.climb_landing_pose + offset, support_id=10)
            self.assertIsNone(surface_action_offset(invalid, self.box, self.capability))

    def test_landing_goal_follows_support_without_mutating_physics_cache(self):
        data = SimpleNamespace(geom_xpos=np.array([[1.0, 2.0, 0.1]]), geom_xmat=np.array([[0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]]))
        before = data.geom_xpos.copy()
        np.testing.assert_allclose(surface_goal(data, 0, np.array([0.06, 0.04])), [0.96, 2.06])
        np.testing.assert_array_equal(data.geom_xpos, before)
        data.geom_xpos[0, 0] += 0.1
        np.testing.assert_allclose(surface_goal(data, 0, np.array([0.06, 0.04])), [1.06, 2.06])

    def test_candidates_are_distinct_repeatable_and_inside_controller_bounds(self):
        observation = OracleObservation(np.zeros(12), np.zeros(4), (self.box, self.platform), self.capability)
        actions = (
            SkillAction(SkillType.PUSH, [1.72, 0.0, 0.0], object_id=10),
            SkillAction(SkillType.NAV, self.box.climb_entry_pose, support_id=10),
            SkillAction(SkillType.CLIMB, self.box.climb_landing_pose, support_id=10),
        )
        for action in actions:
            options = dict(count=7, seed=11, observation=observation, scene_family="box_support_nominal")
            candidates = build_candidates(action, **options)
            repeated = build_candidates(action, **options)
            self.assertEqual(len(candidates), 7)
            self.assertEqual(len({tuple(candidate.action.target_pose.tolist()) for candidate in candidates}), 7)
            self.assertEqual(candidates[0].source, "reference")
            self.assertEqual(candidates[-1].source, "invalid_request")
            for candidate, duplicate in zip(candidates, repeated):
                np.testing.assert_array_equal(candidate.action.target_pose, duplicate.action.target_pose)
            for candidate in candidates[1:-1]:
                if action.skill == SkillType.PUSH:
                    self.assertTrue(push_target_supported(candidate.action.target_pose, self.box, self.platform))
                else:
                    self.assertIsNotNone(surface_action_offset(candidate.action, self.box, self.capability))


if __name__ == "__main__":
    unittest.main()