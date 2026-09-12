"""Physical replay checks for isolated control-boundary snapshots."""

from __future__ import annotations

from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import mujoco
import numpy as np
import torch

from reconfigurable_navigation.climb_runtime import ClimbRuntime
from reconfigurable_navigation.complex_course_env import ComplexCourseEnv
from reconfigurable_navigation.data.snapshot import SimulatorSnapshot
from reconfigurable_navigation.locomotion_runtime import LocomotionRuntime
from reconfigurable_navigation.representations import SkillAction, SkillType
from reconfigurable_navigation.runtime import ReconfigurableExecutor


def integration_state(executor: ReconfigurableExecutor) -> np.ndarray:
    state_type = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(executor.env.model, state_type))
    mujoco.mj_getState(executor.env.model, executor.env.data, state, state_type)
    return state


class SnapshotChecks(unittest.TestCase):
    def setUp(self) -> None:
        env = ComplexCourseEnv()
        env.reset(seed=0)
        runtime = LocomotionRuntime(env)
        climb = ClimbRuntime(model=env.model, data=env.data, deploy_config=env.deploy_cfg)
        self.executor = ReconfigurableExecutor(env, runtime, climb_runtime=climb)
        for _ in range(150):
            runtime.hold_default()
        for _ in range(5):
            runtime.step(np.array([0.25, 0.0, 0.0]))

    def rollout(self, executor: ReconfigurableExecutor, velocity: np.ndarray, steps: int = 100):
        states = []
        actions = []
        for _ in range(steps):
            runtime = executor.climb_runtime if executor.active_runtime == SkillType.CLIMB else executor.runtime
            actions.append(runtime.step(velocity).copy())
            states.append(integration_state(executor))
        return np.asarray(states), np.asarray(actions)

    def test_fork_preserves_state_caches_and_shared_handles(self) -> None:
        snapshot = SimulatorSnapshot.capture(self.executor)
        branch = snapshot.fork()
        self.assertIsNot(branch.env.model, self.executor.env.model)
        self.assertIsNot(branch.env.data, self.executor.env.data)
        self.assertIs(branch.runtime.env, branch.env)
        self.assertIs(branch.runtime.model, branch.env.model)
        self.assertIs(branch.climb_runtime.data, branch.env.data)
        self.assertEqual(branch.env.reset_count, self.executor.env.reset_count)
        np.testing.assert_array_equal(integration_state(branch), integration_state(self.executor))
        np.testing.assert_array_equal(branch.env.data.xpos, self.executor.env.data.xpos)
        np.testing.assert_array_equal(branch.runtime.observation(), self.executor.runtime.observation())
        np.testing.assert_array_equal(branch.climb_runtime.observation(), self.executor.climb_runtime.observation())

    def test_same_snapshot_matches_live_continuation_three_times(self) -> None:
        snapshot = SimulatorSnapshot.capture(self.executor)
        velocity = np.array([0.3, 0.0, 0.1])
        expected_states, expected_actions = self.rollout(self.executor, velocity)
        for _ in range(3):
            states, actions = self.rollout(snapshot.fork(), velocity)
            np.testing.assert_allclose(states, expected_states, rtol=0.0, atol=1.0e-10)
            np.testing.assert_array_equal(actions, expected_actions)

    def test_branch_order_and_mutation_do_not_change_parent(self) -> None:
        snapshot = SimulatorSnapshot.capture(self.executor)
        parent_state = integration_state(self.executor)
        parent_mass = self.executor.env.model.body_mass.copy()
        expected, _actions = self.rollout(snapshot.fork(), np.array([0.25, 0.0, 0.0]))
        other = snapshot.fork()
        other.env.model.body_mass[other.env.box_body_id] *= 2.0
        other.runtime.kp[:] *= 0.5
        other.climb_runtime.actuator_delay_steps[:] = 0
        self.rollout(other, np.array([0.0, 0.15, 0.0]))
        actual, _actions = self.rollout(snapshot.fork(), np.array([0.25, 0.0, 0.0]))
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(integration_state(self.executor), parent_state)
        np.testing.assert_array_equal(self.executor.env.model.body_mass, parent_mass)

    def test_climb_delay_history_replays(self) -> None:
        self.executor._activate_runtime(SkillType.CLIMB)
        for _ in range(5):
            self.executor.climb_runtime.step(np.array([0.4, 0.0, 0.0]))
        snapshot = SimulatorSnapshot.capture(self.executor)
        expected_states, expected_actions = self.rollout(self.executor, np.array([0.4, 0.0, 0.0]))
        for _ in range(3):
            branch = snapshot.fork()
            self.assertEqual(branch.active_runtime, SkillType.CLIMB)
            states, actions = self.rollout(branch, np.array([0.4, 0.0, 0.0]))
            np.testing.assert_allclose(states, expected_states, rtol=0.0, atol=1.0e-10)
            np.testing.assert_array_equal(actions, expected_actions)

    def test_environment_rng_and_policy_weights_are_independent(self) -> None:
        snapshot = SimulatorSnapshot.capture(self.executor)
        first = snapshot.fork()
        second = snapshot.fork()
        np.testing.assert_array_equal(first.env.rng.random(10), second.env.rng.random(10))
        original_parameter = next(self.executor.runtime.policy.parameters())
        branch_parameter = next(first.runtime.policy.parameters())
        self.assertNotEqual(original_parameter.data_ptr(), branch_parameter.data_ptr())
        self.assertNotEqual(branch_parameter.data_ptr(), next(second.runtime.policy.parameters()).data_ptr())

    def test_archive_restores_cached_observations_and_replay(self) -> None:
        snapshot = SimulatorSnapshot.capture(self.executor)
        expected, expected_actions = self.rollout(snapshot.fork(), np.array([0.3, 0.0, 0.1]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.snapshot"
            snapshot.save(path)
            with self.assertRaises(ValueError):
                SimulatorSnapshot.load(path)
            restored = SimulatorSnapshot.load(path, trusted=True)
            np.testing.assert_array_equal(restored.integration_state(), snapshot.integration_state())
            branch = restored.fork()
            np.testing.assert_array_equal(branch.env.data.xpos, self.executor.env.data.xpos)
            np.testing.assert_array_equal(branch.runtime.observation(), self.executor.runtime.observation())
            actual, actual_actions = self.rollout(branch, np.array([0.3, 0.0, 0.1]))
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(actual_actions, expected_actions)
            with self.assertRaises(FileExistsError):
                snapshot.save(path)

    def test_archive_checksum_rejects_corrupted_native_state(self) -> None:
        snapshot = SimulatorSnapshot.capture(self.executor)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.snapshot"
            snapshot.save(path)
            with ZipFile(path) as archive:
                contents = {name: archive.read(name) for name in archive.namelist()}
            contents["model.bin"] = b"broken" + contents["model.bin"][6:]
            corrupt = Path(directory) / "corrupt.snapshot"
            with ZipFile(corrupt, "w") as archive:
                for name, content in contents.items():
                    archive.writestr(name, content)
            with self.assertRaisesRegex(ValueError, "checksum"):
                SimulatorSnapshot.load(corrupt, trusted=True)

    def navigation_action(self, distance: float = 0.3) -> SkillAction:
        position = self.executor.env.observe().robot_state
        return SkillAction(SkillType.NAV, np.array([position[0] + distance, position[1], 0.0]))

    def test_skill_rollout_repeats_without_changing_parent_or_global_rng(self) -> None:
        snapshot = SimulatorSnapshot.capture(self.executor)
        original = ReconfigurableExecutor._execute_skill
        draws = []

        def execute_with_rng(executor, *args, **kwargs):
            draws.append((random.random(), float(np.random.random()), float(torch.rand(()))))
            return original(executor, *args, **kwargs)

        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        parent_state = integration_state(self.executor)
        with patch.object(ReconfigurableExecutor, "_execute_skill", execute_with_rng):
            first = snapshot.rollout(self.navigation_action())
            snapshot.rollout(self.navigation_action(0.15))
            repeated = snapshot.rollout(self.navigation_action())
        self.assertEqual(first.reason, repeated.reason)
        self.assertIsNotNone(first.transition)
        self.assertEqual(first.transition.to_json(), repeated.transition.to_json())
        self.assertEqual(draws[0], draws[1])
        self.assertEqual(draws[0], draws[2])
        self.assertEqual(random.getstate(), python_state)
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
        np.testing.assert_array_equal(torch.get_rng_state(), torch_state)
        np.testing.assert_array_equal(integration_state(self.executor), parent_state)

    def test_unsupported_skill_and_budget_are_not_physical_failure_labels(self) -> None:
        snapshot = SimulatorSnapshot.capture(self.executor)
        rejected = snapshot.rollout(SkillAction(SkillType.JUMP, np.zeros(4)))
        self.assertEqual(rejected.reason, "unsupported_skill")
        self.assertIsNone(rejected.transition)
        truncated = snapshot.rollout(self.navigation_action(), max_control_steps=1)
        self.assertEqual(truncated.reason, "rollout_budget_exhausted")
        self.assertTrue(truncated.transition.interrupted)
        self.assertIsNone(truncated.next_snapshot)

    def test_archived_snapshot_replays_a_whole_skill(self) -> None:
        snapshot = SimulatorSnapshot.capture(self.executor, previous_skill=SkillType.PUSH)
        expected = snapshot.rollout(self.navigation_action())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "skill.snapshot"
            snapshot.save(path)
            restored = SimulatorSnapshot.load(path, trusted=True)
            actual = restored.rollout(self.navigation_action())
            self.assertEqual(actual.transition.to_json(), expected.transition.to_json())
            self.assertEqual(actual.transition.previous_skill, SkillType.PUSH)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()