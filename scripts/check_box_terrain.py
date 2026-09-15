"""Validate the climb-box training geometry without starting Isaac Sim."""

from pathlib import Path
import runpy
from types import SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TERRAIN = ROOT / "source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/mdp/box_terrain.py"
generate = runpy.run_path(str(TERRAIN))["single_box_terrain"]


class BoxTerrainChecks(unittest.TestCase):
    def test_gap_range_changes_support_edge_without_moving_goal_origin(self):
        config = SimpleNamespace(size=(6.0, 6.0), box_length=2.0, box_width=1.6, approach_distance=0.75, box_height_range=(0.20, 0.20), approach_height=0.20, approach_length=1.2, approach_width=1.2, approach_gap=0.015, approach_gap_range=(0.0, 0.07))
        origins = []
        for difficulty, gap in ((0.0, 0.0), (0.5, 0.035), (1.0, 0.07)):
            meshes, origin = generate(difficulty, config)
            self.assertAlmostEqual(meshes[1].bounds[0, 0] - meshes[2].bounds[1, 0], gap)
            origins.append(origin)
        np.testing.assert_allclose(origins, np.broadcast_to(origins[0], (3, 3)))

    def test_following_platform_preserves_first_box_goal_and_height(self):
        config = SimpleNamespace(size=(6.0, 6.0), box_length=1.2, box_width=1.2, approach_distance=0.75, box_height_range=(0.20, 0.20), following_platform_height=0.40, following_platform_length=2.0, following_platform_width=1.6, following_platform_gap=0.02)
        meshes, origin = generate(0.5, config)
        self.assertEqual(len(meshes), 3)
        self.assertAlmostEqual(meshes[1].bounds[1, 2], 0.20)
        self.assertAlmostEqual(meshes[2].bounds[1, 2], 0.40)
        self.assertAlmostEqual(meshes[2].bounds[0, 0] - meshes[1].bounds[1, 0], 0.02)
        self.assertAlmostEqual(meshes[1].bounds[0, 0] - origin[0], 0.75)

    def test_raised_approach_preserves_relative_riser_and_real_gap(self):
        config = SimpleNamespace(size=(6.0, 6.0), box_length=2.0, box_width=1.6, approach_distance=0.75, box_height_range=(0.20, 0.20), approach_height=0.20, approach_length=1.2, approach_width=1.2, approach_gap=0.015)
        meshes, origin = generate(0.5, config)
        self.assertEqual(len(meshes), 3)
        self.assertAlmostEqual(origin[2], 0.20)
        self.assertAlmostEqual(meshes[1].bounds[1, 2], 0.40)
        self.assertAlmostEqual(meshes[2].bounds[1, 2], 0.20)
        self.assertAlmostEqual(meshes[1].bounds[0, 0] - meshes[2].bounds[1, 0], 0.015)
        self.assertTrue(meshes[2].bounds[0, 0] < origin[0] < meshes[2].bounds[1, 0])

    def test_ground_spawn_and_single_box_height(self):
        config = SimpleNamespace(size=(6.0, 6.0), box_length=1.2, box_width=1.2, approach_distance=0.75, box_height_range=(0.05, 0.30))
        for difficulty, height in ((0.0, 0.05), (0.6, 0.20), (1.0, 0.30)):
            meshes, origin = generate(difficulty, config)
            self.assertEqual(len(meshes), 2)
            self.assertEqual(origin[2], 0.0)
            self.assertAlmostEqual(meshes[0].bounds[1, 2], 0.0)
            self.assertAlmostEqual(meshes[1].bounds[1, 2], height)
            self.assertAlmostEqual(meshes[1].bounds[0, 2], 0.0)
            self.assertAlmostEqual(meshes[1].bounds[0, 0] - origin[0], config.approach_distance)
            np.testing.assert_allclose(meshes[1].extents[:2], [1.2, 1.2])


if __name__ == "__main__":
    unittest.main()