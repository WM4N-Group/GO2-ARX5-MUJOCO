"""Contracts for candidate parameters and counterfactual result labels."""

from __future__ import annotations

import json
import unittest

import numpy as np

from reconfigurable_navigation.data.candidates import SkillCandidate, build_candidates, candidate_payload
from reconfigurable_navigation.data.snapshot import SkillReplay
from reconfigurable_navigation.data.transition import SkillTransition
from reconfigurable_navigation.representations import Capability, OracleObservation, SkillAction, SkillType
from reconfigurable_navigation.skills import SkillStatus


class CandidateChecks(unittest.TestCase):
    def setUp(self) -> None:
        self.action = SkillAction(SkillType.PUSH, np.array([1.5, 0.0, 0.0]), object_id=10)
        self.observation = OracleObservation(np.zeros(12), np.zeros(4), (), Capability())

    def transition(self, interrupted: bool = False) -> SkillTransition:
        return SkillTransition(
            self.action, self.observation, self.observation,
            SkillStatus.FAILED, 1, 0.0, 0.02, 0.02, interrupted=interrupted,
        )

    def test_candidates_are_reproducible_and_do_not_mutate_reference(self) -> None:
        original = self.action.target_pose.copy()
        first = build_candidates(self.action, count=12, seed=5)
        second = build_candidates(self.action, count=12, seed=5)
        self.assertEqual(len(first), 12)
        self.assertEqual([candidate.source for candidate in first], [candidate.source for candidate in second])
        for candidate, repeated in zip(first, second):
            np.testing.assert_array_equal(candidate.action.target_pose, repeated.action.target_pose)
        first[0].action.target_pose[:] = -99.0
        np.testing.assert_array_equal(self.action.target_pose, original)
        self.assertEqual(first[-1].action.object_id, -1)

    def test_climb_perturbations_preserve_target_shape_and_support(self) -> None:
        action = SkillAction(SkillType.CLIMB, np.array([0.8, 4.7, 0.41, np.pi / 2]), support_id=20)
        candidates = build_candidates(action, count=6, seed=0)
        self.assertEqual(candidates[-1].source, "invalid_request")
        for candidate in candidates[:-1]:
            self.assertEqual(candidate.action.target_pose.shape, (4,))
            self.assertTrue(np.isfinite(candidate.action.target_pose).all())
            self.assertEqual(candidate.action.support_id, 20)
        self.assertFalse(np.array_equal(candidates[2].action.target_pose, action.target_pose))

    def test_rejection_does_not_fabricate_a_transition(self) -> None:
        payload = candidate_payload(SkillCandidate("invalid_request", self.action), SkillReplay(None, None, "skill_not_executable"))
        self.assertFalse(payload["executed"])
        self.assertEqual(payload["outcome"], "rejected")
        self.assertIsNone(payload["transition"])
        self.assertIsNone(payload["skill_success"])
        self.assertFalse(payload["label_validity"]["dynamics"])
        json.dumps(payload, allow_nan=False)

    def test_budget_truncation_masks_success_label(self) -> None:
        payload = candidate_payload(SkillCandidate("reference", self.action), SkillReplay(self.transition(True), None, "rollout_budget_exhausted"))
        self.assertTrue(payload["executed"])
        self.assertTrue(payload["truncated"])
        self.assertEqual(payload["outcome"], "truncated")
        self.assertIsNone(payload["skill_success"])
        self.assertFalse(payload["label_validity"]["skill_success"])

    def test_executed_failure_is_preserved(self) -> None:
        payload = candidate_payload(SkillCandidate("boundary_parameter", self.action), SkillReplay(self.transition(), None, "failed"))
        self.assertTrue(payload["executed"])
        self.assertFalse(payload["truncated"])
        self.assertIs(payload["skill_success"], False)
        self.assertTrue(payload["label_validity"]["skill_success"])
        self.assertFalse(payload["label_validity"]["reachability"])

    def test_unsupported_reference_skill_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_candidates(SkillAction(SkillType.JUMP, np.zeros(4)), count=5, seed=0)


if __name__ == "__main__":
    unittest.main()