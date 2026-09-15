"""Check real model geometry and deterministic scene-family assignments."""

from dataclasses import replace
import unittest

import mujoco
import numpy as np

from reconfigurable_navigation.box_support_env import make_model
from reconfigurable_navigation.box_support_scene import BoxSupportScene, box_support_suite
from reconfigurable_navigation.data.candidates import candidate_partition_metadata


class SceneChecks(unittest.TestCase):
    def test_default_scene_preserves_original_model_arrays(self):
        original = make_model(platform_height=0.4)
        configured = make_model(scene=BoxSupportScene())
        for name in ("qpos0", "geom_pos", "geom_size", "geom_friction", "body_mass", "body_inertia", "body_pos", "body_quat"):
            np.testing.assert_array_equal(getattr(original, name), getattr(configured, name))

    def test_scene_parameters_change_actual_physics(self):
        scene = BoxSupportScene(box_pose=(1.2, 0.1, 0.03), box_size=(1.1, 1.0, 0.18), box_mass=4.0, box_friction=0.5, platform_pose=(3.4, 0.1, 0.0), platform_size=(1.8, 1.5, 0.38))
        model = make_model(scene=scene)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        box, platform = model.geom("push_box").id, model.geom("high_platform").id
        np.testing.assert_allclose(model.geom_size[box] * 2.0, scene.box_size)
        np.testing.assert_allclose(model.geom_size[platform] * 2.0, scene.platform_size)
        np.testing.assert_allclose(data.geom_xpos[platform], [3.4, 0.1, 0.19])
        self.assertEqual(model.body_mass[model.geom_bodyid[box]], 4.0)
        self.assertEqual(model.geom_friction[box, 0], 0.5)

    def test_scene_ids_and_family_splits_are_stable(self):
        scenes = box_support_suite()
        self.assertEqual(len({scene.scene_id for scene in scenes.values()}), 6)
        assignments = {}
        for scene in scenes.values():
            self.assertEqual(scene.scene_id, BoxSupportScene(**scene.to_dict()).scene_id)
            self.assertEqual(assignments.setdefault(scene.scene_family, scene.dataset_split), scene.dataset_split)
        self.assertEqual(set(assignments.values()), {"train", "validation", "test"})

    def test_invalid_geometry_is_rejected_before_model_creation(self):
        for fields in ({"box_mass": -1.0}, {"platform_pose": (1.2, 0.0, 0.0)}, {"platform_pose": (3.3, 0.0, 0.2)}, {"box_friction": np.nan}):
            with self.assertRaises(ValueError):
                replace(BoxSupportScene(), **fields)

    def test_partition_rejects_family_leakage_and_mixed_legacy_data(self):
        record = {"metadata": {"scene_family": "box_support_aligned", "dataset_split": "train", "split_group_id": "aligned"}}
        self.assertEqual(candidate_partition_metadata([record])["split"], "scene_family_v1")
        conflicting = {"metadata": {**record["metadata"], "dataset_split": "test"}}
        with self.assertRaisesRegex(ValueError, "crosses"):
            candidate_partition_metadata([record, conflicting])
        with self.assertRaisesRegex(ValueError, "mix"):
            candidate_partition_metadata([record, {"metadata": {"scene_family": "box_support_nominal"}}])


if __name__ == "__main__":
    unittest.main()