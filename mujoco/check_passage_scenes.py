"""Check parameterized passage geometry and physical state contracts."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from evaluate_passage_scenes import run_case
from reconfigurable_navigation.data.snapshot import SimulatorSnapshot
from reconfigurable_navigation.env import BlockedPassageEnv
from reconfigurable_navigation.locomotion_runtime import LocomotionRuntime
from reconfigurable_navigation.passage_scene import ParameterizedPassageEnv, PassageScene, passage_sweep
from reconfigurable_navigation.representations import SkillAction, SkillType
from reconfigurable_navigation.runtime import ReconfigurableExecutor


class PassageSceneChecks(unittest.TestCase):
    def test_compiled_box_mass_inertia_and_truth_match_parameters(self) -> None:
        scene = PassageScene(corridor_width=2.0, box_size=(0.50, 1.30, 0.40), box_mass=7.0, box_friction=0.03)
        env = ParameterizedPassageEnv(scene)
        observation = env.reset(seed=0)
        box = observation.objects[-1]
        np.testing.assert_allclose(box.size, scene.box_size)
        self.assertEqual(box.mass, scene.box_mass)
        self.assertEqual(env.model.geom_friction[env.box_geom_id, 0], scene.box_friction)
        size_squared = np.square(scene.box_size)
        expected_inertia = scene.box_mass / 12.0 * (size_squared.sum() - size_squared)
        np.testing.assert_allclose(env.model.body_inertia[env.box_body_id], expected_inertia)
        np.testing.assert_allclose(observation.objects[0].center, [0.0, 1.1, 0.3])
        np.testing.assert_allclose(observation.objects[2].size, [0.2, 2.4, 0.6])
        self.assertEqual(env.reset_count, 1)

    def test_default_sample_reproduces_legacy_initial_physics(self) -> None:
        original = BlockedPassageEnv()
        original.reset(seed=4)
        variant = ParameterizedPassageEnv(passage_sweep(4)["baseline"])
        variant.reset(seed=4)
        np.testing.assert_array_equal(variant.data.qpos, original.data.qpos)
        for name in ("body_mass", "body_inertia", "geom_size", "geom_friction", "body_pos"):
            np.testing.assert_array_equal(getattr(variant.model, name), getattr(original.model, name))
        np.testing.assert_array_equal(variant.observe().robot_state, original.observe().robot_state)

    def test_parameters_round_trip_and_sampling_are_deterministic(self) -> None:
        first = passage_sweep(7)
        repeated = passage_sweep(7)
        self.assertEqual(first, repeated)
        self.assertEqual(len({scene.scene_id for scene in first.values()}), len(first))
        for scene in first.values():
            restored = PassageScene(**json.loads(json.dumps(scene.to_dict())))
            self.assertEqual(restored, scene)
            self.assertEqual(restored.scene_id, scene.scene_id)
        self.assertNotEqual(first["baseline"].scene_id, passage_sweep(8)["baseline"].scene_id)

    def test_invalid_or_intersecting_scene_parameters_are_rejected(self) -> None:
        for update in (
            {"box_mass": -1.0}, {"box_friction": float("nan")},
            {"box_size": (0.4, 2.0, 0.6)}, {"box_pose": (0.0, 0.2, 0.0)},
            {"robot_pose": (5.0, 0.0, 0.0)}, {"goal_pose": (1.0, 0.0)},
        ):
            with self.subTest(update=update), self.assertRaises(ValueError):
                PassageScene(**update)
        env = ParameterizedPassageEnv(PassageScene(robot_pose=(0.0, 0.0, 0.0)))
        with self.assertRaisesRegex(ValueError, "collides"):
            env.reset(0)

    def test_configured_poses_and_model_instances_are_independent(self) -> None:
        scene = PassageScene(box_size=(0.4, 1.2, 0.4), box_pose=(0.1, 0.0, 0.04), goal_pose=(1.6, 0.0, 0.1))
        first = ParameterizedPassageEnv(scene)
        second = ParameterizedPassageEnv(replace(scene, box_mass=10.0))
        observation = first.reset(1)
        second.reset(1)
        np.testing.assert_allclose(observation.objects[-1].center, [0.1, 0.0, 0.2])
        self.assertAlmostEqual(observation.objects[-1].yaw, 0.04)
        np.testing.assert_allclose(observation.goal[:3], scene.goal_pose)
        self.assertEqual(first.model.body_mass[first.box_body_id], 5.0)
        self.assertEqual(second.model.body_mass[second.box_body_id], 10.0)

    def test_parameterized_snapshot_retains_scene_and_skill_replay(self) -> None:
        scene = passage_sweep(0)["mass_low"]
        env = ParameterizedPassageEnv(scene)
        env.reset(0)
        runtime = LocomotionRuntime(env)
        executor = ReconfigurableExecutor(env, runtime)
        for _ in range(150):
            runtime.hold_default()
        snapshot = SimulatorSnapshot.capture(executor)
        action = SkillAction(SkillType.NAV, np.array([-2.0, 0.0, 0.0]))
        expected = snapshot.rollout(action)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "passage.snapshot"
            snapshot.save(path)
            restored = SimulatorSnapshot.load(path, trusted=True)
            branch = restored.fork()
            self.assertEqual(branch.env.scene, scene)
            self.assertEqual(branch.env.reset_count, 1)
            self.assertEqual(branch.env.observe().objects[-1].mass, 3.0)
            actual = restored.rollout(action)
            self.assertEqual(actual.transition.to_json(), expected.transition.to_json())
            self.assertEqual(env.reset_count, 1)

    def test_rejected_scene_has_no_fabricated_skill_or_contact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = run_case(
                passage_sweep(0)["mass_over_limit"], 0, "mass_over_limit", output,
                snapshots=False, group_id="same-layout",
            )
            self.assertEqual(result["reason"], "blocking_object_too_heavy")
            self.assertEqual(result["records"], 0)
            self.assertIsNone(result["push_fingertip_contact"])
            self.assertFalse((output / "transitions.jsonl").exists())


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()