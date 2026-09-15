"""World-model encoding and leakage checks without external policy artifacts."""

from copy import deepcopy
from dataclasses import asdict
import unittest
import tempfile
from pathlib import Path
import hashlib
import json

import numpy as np
import torch

from reconfigurable_navigation.representations import Capability, ObjectState, ObjectType, OracleObservation, SkillAction, SkillType
from reconfigurable_navigation.world_model.encoding import FEATURE_NAMES, encode_example, encode_inputs
from reconfigurable_navigation.world_model.model import SkillWorldModel, fit_normalization
from reconfigurable_navigation.world_model.evaluation import _classification
from reconfigurable_navigation.world_model.dataset import load_dataset


def example():
    robot = np.zeros(12)
    robot[2] = 0.33
    robot[11] = 1.0
    objects = (ObjectState(10, ObjectType.MOVABLE_BOX, np.array([1.1, 0.0, 0.1]), np.array([1.2, 1.2, 0.2]), movable=True, supportable=True, mass=5.0), ObjectState(20, ObjectType.PLATFORM, np.array([3.3, 0.0, 0.2]), np.array([2.0, 1.6, 0.4]), supportable=True))
    before = asdict(OracleObservation(robot, np.array([3.3, 0.0, 0.4, 1.0]), objects, Capability()))
    after = deepcopy(before)
    after["robot_state"][0] = 0.5
    after["objects"][0]["center"][0] = 1.65
    candidate = {
        "executed": True, "truncated": False,
        "action": asdict(SkillAction(SkillType.PUSH, [1.72, 0.0, 0.0], object_id=10)),
        "transition": {"observation_before": before, "observation_after": after, "previous_skill": None, "elapsed_sim_time": 5.0, "interrupted": False},
        "skill_success": True, "process_labels": {"illegal_collision": False},
        "label_validity": {"dynamics": True, "skill_success": True, "illegal_collision": True},
        "continuation": {"budget": {"max_control_steps": 3000, "max_skills": 8}, "oracle_task_success": True, "total_sim_time": 30.0, "label_validity": {"total_cost": True, "oracle_task_success": True}},
    }
    source = {"metadata": {"scene_parameters": {"box_friction": 0.4}, "start_context": {"completed_operations": []}}}
    return candidate, source


def write_shard(directory, split):
    directory.mkdir()
    candidate, source = example()
    candidate["transition"]["action"] = deepcopy(candidate["action"])
    context = {"dataset_split": split, "scene_family": "box_support_aligned", "scene_id": "same-scene", "episode_id": "same-episode", "snapshot_sha256": "snapshot", "split_group_id": "aligned"}
    source.update(deepcopy(candidate["transition"]))
    source["metadata"].update(context, policy_sha256={"push": "frozen"})
    candidate.update(context, candidate_id=f"candidate-{split}", source_record_index=0, snapshot_code_sha256="same-runtime", outcome="succeeded")
    source_raw = (json.dumps(source, default=lambda value: value.tolist()) + "\n").encode()
    candidate_raw = (json.dumps(candidate, default=lambda value: value.tolist()) + "\n").encode()
    (directory / "source.jsonl").write_bytes(source_raw)
    (directory / "candidates.jsonl").write_bytes(candidate_raw)
    manifest = {"schema_version": 2, "complete": True, "split": "scene_family_v1", "records": 1, "source_jsonl": "source.jsonl", "source_sha256": hashlib.sha256(source_raw).hexdigest(), "candidates_sha256": hashlib.sha256(candidate_raw).hexdigest()}
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


class EncodingChecks(unittest.TestCase):
    def test_inputs_do_not_leak_results_or_split_identifiers(self):
        candidate, source = example()
        expected = encode_example(candidate, source)[0]
        candidate["transition"]["observation_after"]["robot_state"][:] += 3.0
        candidate["skill_success"] = False
        candidate["continuation"].update(oracle_task_success=False, total_sim_time=50.0)
        source["metadata"].update(dataset_split="test", scene_seed=999, scene_id="different", episode_id="other")
        np.testing.assert_array_equal(encode_example(candidate, source)[0], expected)
        self.assertEqual(expected.shape, (len(FEATURE_NAMES),))

    def test_inputs_are_invariant_to_horizontal_world_translation(self):
        candidate, source = example()
        expected = encode_example(candidate, source)[0]
        candidate["action"]["target_pose"][:2] += [2.0, -3.0]
        for observation in (candidate["transition"]["observation_before"], candidate["transition"]["observation_after"]):
            observation["robot_state"][:2] += [2.0, -3.0]
            observation["goal"][:2] += [2.0, -3.0]
            for obj in observation["objects"]:
                obj["center"][:2] += [2.0, -3.0]
        np.testing.assert_allclose(encode_example(candidate, source)[0], expected, atol=1e-6)

    def test_rejected_requests_have_no_dynamics_sample(self):
        candidate, source = example()
        candidate.update(executed=False, transition=None)
        self.assertIsNone(encode_example(candidate, source))

    def test_truncation_masks_terminal_targets_but_keeps_positive_collision(self):
        candidate, source = example()
        candidate["truncated"] = candidate["transition"]["interrupted"] = True
        candidate["skill_success"] = None
        candidate["process_labels"]["illegal_collision"] = True
        candidate["continuation"].update(oracle_task_success=None, label_validity={"total_cost": False, "oracle_task_success": False})
        _features, _regression, regression_mask, _binary, binary_mask = encode_example(candidate, source)
        self.assertFalse(regression_mask.any())
        np.testing.assert_array_equal(binary_mask, [False, True, False])

    def test_unknown_task_is_not_a_negative_label(self):
        candidate, source = example()
        candidate["continuation"] = {}
        _features, regression, regression_mask, _binary, binary_mask = encode_example(candidate, source)
        np.testing.assert_allclose(regression[:3], [0.5, 0.0, 0.0])
        np.testing.assert_allclose(regression[5:8], [0.55, 0.0, 0.0])
        self.assertFalse(regression_mask[-1])
        self.assertFalse(binary_mask[-1])

    def test_normalization_uses_training_split_only(self):
        encoded = encode_example(*example())
        dataset = {name: np.stack([values, values, values]) for name, values in zip(("features", "regression", "regression_mask", "binary", "binary_mask"), encoded)}
        dataset["split"] = np.array(["train", "validation", "test"])
        expected = fit_normalization(dataset)
        dataset["features"][1:] += 100.0
        dataset["regression"][1:] -= 200.0
        for name, values in fit_normalization(dataset).items():
            np.testing.assert_array_equal(values, expected[name])

    def test_masked_labels_do_not_change_loss(self):
        torch.manual_seed(0)
        model = SkillWorldModel()
        tensors = [torch.from_numpy(np.stack([value])) for value in encode_example(*example())]
        tensors[2].fill_(False)
        tensors[4].fill_(False)
        self.assertEqual(float(model.loss(*tensors)), 0.0)
        tensors[1].add_(100.0)
        tensors[3].fill_(1.0)
        self.assertEqual(float(model.loss(*tensors)), 0.0)

    def test_checkpoint_roundtrip_preserves_predictions(self):
        model = SkillWorldModel().eval()
        features = torch.from_numpy(encode_example(*example())[0]).unsqueeze(0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            model.save(path, {"scope": "unit-test"})
            restored, metadata = SkillWorldModel.load(path)
            self.assertEqual(metadata["scope"], "unit-test")
            for actual, expected in zip(restored(features), model(features)):
                torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_classification_metrics_cover_ties_and_single_class(self):
        labels = np.array([0.0, 1.0])
        self.assertEqual(_classification(labels, np.array([0.5, 0.5]))["auroc"], 0.5)
        self.assertEqual(_classification(labels, np.array([0.1, 0.9]))["auroc"], 1.0)
        self.assertEqual(_classification(labels, np.array([0.9, 0.1]))["auroc"], 0.0)
        self.assertIsNone(_classification(np.ones(2), np.array([0.2, 0.8]))["auroc"])

    def test_dataset_rejects_corruption_and_cross_split_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = write_shard(root / "train", "train")
            test = write_shard(root / "test", "test")
            self.assertEqual(load_dataset([train])["features"].shape[0], 1)
            with self.assertRaisesRegex(ValueError, "crosses"):
                load_dataset([train, test])
            (root / "train/candidates.jsonl").write_text("corrupted")
            with self.assertRaisesRegex(ValueError, "Checksum"):
                load_dataset([train])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()