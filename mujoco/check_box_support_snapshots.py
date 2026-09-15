"""Check independent snapshots for the accepted box-support execution backend."""

import argparse
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from reconfigurable_navigation.data.snapshot import SimulatorSnapshot
from reconfigurable_navigation.representations import SkillAction
from reconfigurable_navigation.runtime.box_support_backend import make_box_support_executor


class BoxSnapshotChecks(unittest.TestCase):
    policies = None

    def setUp(self):
        self.executor = make_box_support_executor(*self.policies, seed=500)

    def test_fork_preserves_controller_graph_and_independent_policies(self):
        snapshot = SimulatorSnapshot.capture(self.executor)
        first, second = snapshot.fork(), snapshot.fork()
        self.assertIs(first.runtime.physics, first.skill_backend.push)
        self.assertIs(first.env.push_runtime, first.skill_backend.push)
        self.assertIs(first.skill_backend.env, first.env)
        self.assertIs(first.skill_backend.active_physics, first.skill_backend.push)
        self.assertIsNot(first.env.model, self.executor.env.model)
        self.assertIsNot(first.env.data, self.executor.env.data)
        for runtime in (first.runtime, *first.skill_backend.snapshot_runtimes()):
            self.assertIs(runtime.model, first.env.model)
            self.assertIs(runtime.data, first.env.data)
        for executor in (first, second):
            parameter = next(executor.skill_backend.push.policy.parameters())
            self.assertNotEqual(parameter.data_ptr(), next(self.executor.skill_backend.push.policy.parameters()).data_ptr())
        self.assertNotEqual(next(first.skill_backend.push.policy.parameters()).data_ptr(), next(second.skill_backend.push.policy.parameters()).data_ptr())
        first.skill_backend.push.arm.goal[:] += 1.0
        np.testing.assert_array_equal(second.skill_backend.push.arm.goal, self.executor.skill_backend.push.arm.goal)
        np.testing.assert_array_equal(snapshot.integration_state(), SimulatorSnapshot.capture(self.executor).integration_state())
        self.assertEqual(first.env.reset_count, 1)

    def test_archive_restores_ik_history_policies_and_planner_aliases(self):
        for _ in range(25):
            self.executor.skill_backend.push.step()
        snapshot = SimulatorSnapshot.capture(self.executor)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "box.snapshot"
            snapshot.save(path)
            restored = SimulatorSnapshot.load(path, trusted=True)
            first, second = snapshot.fork(), restored.fork()
            self.assertIs(second.replanner.planner.completed, second.skill_backend.completed)
            self.assertIs(second.skill_backend.push.arm.runtime, second.skill_backend.push)
            self.assertIs(second.skill_backend.push.arm.model, second.env.model)
            np.testing.assert_array_equal(restored.integration_state(), snapshot.integration_state())
            for _ in range(25):
                np.testing.assert_array_equal(first.skill_backend.push.step(), second.skill_backend.push.step())
                np.testing.assert_array_equal(first.env.data.qpos, second.env.data.qpos)
                np.testing.assert_array_equal(first.env.data.qvel, second.env.data.qvel)
                np.testing.assert_array_equal(first.skill_backend.push.arm.reference, second.skill_backend.push.arm.reference)
                np.testing.assert_array_equal(first.skill_backend.push.target_history, second.skill_backend.push.target_history)

    def test_active_callback_is_rejected_at_snapshot_capture(self):
        self.executor.skill_backend.push.on_step = lambda runtime: None
        with self.assertRaisesRegex(ValueError, "boundary"):
            SimulatorSnapshot.capture(self.executor)

    def test_rejection_and_budget_leave_live_world_unchanged(self):
        snapshot = SimulatorSnapshot.capture(self.executor)
        action = self.executor.replanner.decide(self.executor.env.observe()).action
        rejected = snapshot.rollout(SkillAction(action.skill, action.target_pose + [0.01, 0.0, 0.0], action.object_id, action.support_id))
        self.assertIsNone(rejected.transition)
        truncated = snapshot.rollout(action, max_control_steps=1)
        record = json.loads(truncated.transition.to_json())
        self.assertEqual(truncated.reason, "rollout_budget_exhausted")
        self.assertTrue(record["interrupted"])
        self.assertIsNone(record["skill_success"])
        self.assertIsNone(record["failure_reason"])
        self.assertIsNone(truncated.next_snapshot)
        np.testing.assert_array_equal(snapshot.integration_state(), SimulatorSnapshot.capture(self.executor).integration_state())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--push-policy", type=Path, required=True)
    parser.add_argument("--climb-policy", type=Path, required=True)
    parser.add_argument("--platform-policy", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    BoxSnapshotChecks.policies = (args.push_policy, args.climb_policy, args.platform_policy)
    torch.set_num_threads(1)
    unittest.main(argv=[__file__, *remaining])