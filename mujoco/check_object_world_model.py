"""Check spatial encodings and object-model contracts."""

from copy import deepcopy
import unittest
import tempfile
from pathlib import Path

import numpy as np
import torch

import check_world_model as tabular_checks
from replay_skill_records import replay_record
from reconfigurable_navigation.world_model.spatial_encoding import BEV_RESOLUTION, BEV_SIZE, MAX_OBJECTS, OBJECT_TOKEN_NAMES, PROPRIO_FIELDS, encode_proprio, encode_scene
from reconfigurable_navigation.world_model.object_model import ObjectWorldModel
from reconfigurable_navigation.world_model.evaluation import conditional_baseline
from reconfigurable_navigation.world_model.dataset import snapshot_identity


def scene_input():
    candidate, source = tabular_checks.example()
    return candidate["transition"]["observation_before"], candidate["action"], source["metadata"]["scene_parameters"]


def pixel(position):
    return int(np.floor(position / BEV_RESOLUTION + BEV_SIZE / 2.0))


class SpatialChecks(unittest.TestCase):
    def test_shapes_and_physical_height(self):
        encoded = encode_scene(*scene_input())
        self.assertEqual(encoded["bev"].shape, (8, 128, 128))
        self.assertEqual(encoded["objects"].shape, (MAX_OBJECTS, len(OBJECT_TOKEN_NAMES)))
        self.assertEqual(int(encoded["object_mask"].sum()), 2)
        self.assertAlmostEqual(float(encoded["bev"][1, pixel(0.0), pixel(1.1)]), 0.2)
        self.assertAlmostEqual(float(encoded["bev"][1, pixel(0.0), pixel(2.7)]), 0.4)
        self.assertTrue(encoded["bev"][6].all())
        self.assertFalse(encoded["objects"][2:].any())

    def test_geometry_moves_and_rotates_without_mutating_input(self):
        observation, action, parameters = scene_input()
        before = deepcopy(observation)
        original = encode_scene(observation, action, parameters)
        np.testing.assert_array_equal(observation["objects"][0]["center"], before["objects"][0]["center"])
        observation["objects"][0]["center"][0] += 0.4
        moved = encode_scene(observation, action, parameters)
        self.assertGreater(np.count_nonzero(original["bev"][0] != moved["bev"][0]), 0)
        observation["objects"][0]["size"][1] = 0.4
        center = observation["objects"][0]["center"][0]
        horizontal = encode_scene(observation, action, parameters)
        observation["objects"][0]["yaw"] = np.pi / 2.0
        vertical = encode_scene(observation, action, parameters)
        self.assertEqual(horizontal["bev"][0, pixel(0.0), pixel(center + 0.4)], 1.0)
        self.assertEqual(vertical["bev"][0, pixel(0.0), pixel(center + 0.4)], 0.0)
        self.assertEqual(vertical["bev"][0, pixel(0.4), pixel(center)], 1.0)

    def test_permutation_preserves_map_and_object_identity(self):
        observation, action, parameters = scene_input()
        expected = encode_scene(observation, action, parameters)
        observation["objects"] = list(reversed(observation["objects"]))
        actual = encode_scene(observation, action, parameters)
        np.testing.assert_array_equal(actual["bev"], expected["bev"])
        np.testing.assert_array_equal(actual["objects"][:2], expected["objects"][:2][::-1])
        self.assertEqual(actual["object_ids"][:2].tolist(), [20, 10])

    def test_missing_pointers_and_duplicate_ids_are_rejected(self):
        observation, action, parameters = scene_input()
        with self.assertRaises(ValueError):
            encode_scene(observation, {**action, "object_id": 99}, parameters)
        observation["objects"][1]["object_id"] = 10
        with self.assertRaises(ValueError):
            encode_scene(observation, action, parameters)

    def test_encoding_uses_only_the_start_observation(self):
        candidate, source = tabular_checks.example()
        parameters = source["metadata"]["scene_parameters"]
        expected = encode_scene(candidate["transition"]["observation_before"], candidate["action"], parameters)
        candidate["transition"]["observation_after"]["robot_state"][:] += 4.0
        candidate["skill_success"] = False
        actual = encode_scene(candidate["transition"]["observation_before"], candidate["action"], parameters)
        np.testing.assert_array_equal(actual["bev"], expected["bev"])
        np.testing.assert_array_equal(actual["objects"], expected["objects"])

    def test_proprioception_has_explicit_availability(self):
        self.assertFalse(encode_proprio({}).any())
        context = {"proprio_schema": "box_runtime_joint_state_v1", **{name: np.arange(18).tolist() for name in PROPRIO_FIELDS}}
        values = encode_proprio(context)
        self.assertEqual(values.shape, (73,))
        self.assertEqual(values[-1], 1.0)
        np.testing.assert_array_equal(values[:18], np.arange(18))

    def test_memory_identity_is_not_an_archive_checksum(self):
        self.assertEqual(snapshot_identity({"snapshot_id": "state"}), "boundary:state")
        self.assertEqual(snapshot_identity({"snapshot_sha256": "file"}), "archive:file")
        with self.assertRaises(ValueError):
            snapshot_identity({})

    def test_memory_boundary_cannot_be_replayed_as_an_archive(self):
        record = {"schema_version": 2, "metadata": {"full_snapshot_available": False}}
        with self.assertRaisesRegex(ValueError, "no persisted physics archive"):
            replay_record(record, Path("records.jsonl"))


class ObjectModelChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.model = ObjectWorldModel(width=32, layers=2).eval()
        for head in (self.model.robot_head, self.model.box_head, self.model.time_head, self.model.binary_head):
            torch.nn.init.normal_(head.weight, std=0.03)
        candidate, source = tabular_checks.example()
        encoded = tabular_checks.encode_example(candidate, source)
        spatial = encode_scene(*scene_input())
        self.inputs = [torch.from_numpy(encoded[0]).unsqueeze(0), *(torch.from_numpy(spatial[name]).unsqueeze(0) for name in ("bev", "objects", "object_mask", "object_ids"))]

    def test_predictions_are_invariant_to_object_permutation_and_padding(self):
        with torch.inference_mode():
            expected = self.model(*self.inputs)
            changed = [value.clone() for value in self.inputs]
            permutation = torch.arange(MAX_OBJECTS)
            permutation[:2] = torch.tensor([1, 0])
            for index in (2, 3, 4):
                changed[index] = changed[index][:, permutation]
            changed[2][:, 2:] = float("nan")
            changed[4][:, 2:] = 10
            for actual, reference in zip(self.model(*changed), expected):
                torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)

    def test_real_object_features_influence_predictions(self):
        with torch.inference_mode():
            expected = self.model(*self.inputs)[0]
            changed = [value.clone() for value in self.inputs]
            changed[2][:, 0, 0] += 0.5
            self.assertGreater(float((self.model(*changed)[0] - expected).abs().max()), 1e-5)

    def test_checkpoint_reconstructs_the_same_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "object.pt"
            self.model.save(path, {"scope": "test"})
            restored, metadata = ObjectWorldModel.load(path)
            self.assertEqual(metadata["scope"], "test")
            with torch.inference_mode():
                for actual, expected in zip(restored(*self.inputs), self.model(*self.inputs)):
                    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_priors_and_normalization_only_use_training_data(self):
        encoded = tabular_checks.encode_example(*tabular_checks.example())
        spatial = encode_scene(*scene_input())
        data = {name: np.stack([value, value]) for name, value in zip(("features", "regression", "regression_mask", "binary", "binary_mask"), encoded)}
        data.update({name: np.stack([value, value]) for name, value in spatial.items()})
        data.update(split=np.array(["train", "test"]), group=np.array([2, 2]), proprio=np.ones((2, 73), dtype=np.float32))
        model = ObjectWorldModel(width=32, layers=2).eval()
        model.set_normalization(data)
        original = {name: value.clone() for name, value in model.named_buffers()}
        inputs = [torch.from_numpy(data[name]) for name in ("features", "bev", "objects", "object_mask", "object_ids", "proprio")]
        with torch.inference_mode():
            predicted, logits = model(*inputs)
        regression, probabilities = conditional_baseline(data)
        np.testing.assert_array_equal(predicted.numpy(), regression)
        np.testing.assert_allclose(torch.sigmoid(logits).numpy(), probabilities, atol=1e-7)
        for name in ("features", "regression", "objects", "proprio"):
            data[name][1] += 100.0
        data["binary"][1] = 0.0
        model.set_normalization(data)
        for name, value in model.named_buffers():
            torch.testing.assert_close(value, original[name], rtol=0.0, atol=0.0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()