"""Check model transfer without loading the simulator or optimizer."""

import math
from pathlib import Path
import runpy
import tempfile
import unittest

import torch
from torch import nn

helpers = runpy.run_path(str(Path(__file__).resolve().parent / "rsl_rl/initialization.py"))
initialize_models = helpers["initialize_models"]
actor_from_state_dict = helpers["actor_from_state_dict"]


class Actor(nn.Module):
    def __init__(self, inputs, outputs=2):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(inputs, 4), nn.ELU(), nn.Linear(4, outputs))
        self.distribution = nn.Module()
        self.distribution.register_parameter("log_std_param", nn.Parameter(torch.zeros(outputs)))

    def forward(self, values):
        return self.mlp(values)


class InitializationChecks(unittest.TestCase):
    def test_sequential_export_preserves_actor_outputs(self):
        source = Actor(5, 3)
        target = Actor(5, 3)
        inputs = torch.randn(8, 5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "actor.pt"
            torch.jit.script(source.mlp).save(str(path))
            initialize_models(target, None, actor_path=path)
        torch.testing.assert_close(target(inputs), source(inputs))

    def test_nav_to_climb_projection_preserves_repeated_state_outputs(self):
        source = Actor(210, 18)
        target = Actor(253, 18)
        observations = torch.randn(16, 253)
        navigation = torch.cat((
            (observations[:, 3:6] * 0.2).repeat(1, 3),
            observations[:, 6:9].repeat(1, 3),
            observations[:, 12:30].repeat(1, 3),
            (observations[:, 30:48] * 0.05).repeat(1, 3),
            observations[:, 48:66].repeat(1, 3),
            observations[:, 9:12].repeat(1, 3),
            torch.tensor((0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0)).repeat(16, 3),
        ), dim=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "actor.pt"
            torch.jit.trace(source, navigation).save(str(path))
            initialize_models(target, None, actor_path=path, observation_layout="nav_to_climb")
        torch.testing.assert_close(target(observations), source(navigation))
        changed = observations.clone()
        changed[:, :3] += 10.0
        changed[:, 66:] += 10.0
        torch.testing.assert_close(target(changed), target(observations))

    def test_nav_to_climb_rejects_wrong_observation_dimensions(self):
        source = Actor(3, 18)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "actor.pt"
            torch.jit.trace(source, torch.zeros(1, 3)).save(str(path))
            with self.assertRaisesRegex(ValueError, "210 -> 253"):
                initialize_models(Actor(253, 18), None, actor_path=path, observation_layout="nav_to_climb")

    def test_explicit_action_selection_preserves_selected_outputs(self):
        source = Actor(3, 4)
        target = Actor(5, 2)
        inputs = torch.randn(8, 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "actor.pt"
            torch.jit.trace(source, inputs).save(str(path))
            initialize_models(target, None, actor_path=path, action_indices=[0, 2])
        extended = torch.cat((inputs, torch.randn(8, 2)), dim=1)
        torch.testing.assert_close(target(extended), source(inputs)[:, [0, 2]])

    def test_direct_inference_preserves_model_outputs(self):
        source = Actor(5)
        restored = actor_from_state_dict(source.state_dict())
        inputs = torch.randn(8, 5)
        torch.testing.assert_close(restored(inputs), source(inputs))
        with self.assertRaises(ValueError):
            actor_from_state_dict({**source.state_dict(), "obs_normalizer.mean": torch.zeros(5)})

    def test_extra_features_preserve_pretrained_actor_output(self):
        source = Actor(3)
        target = Actor(5)
        inputs = torch.randn(8, 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "actor.pt"
            torch.jit.trace(source, inputs).save(str(path))
            initialize_models(target, None, actor_path=path, initial_std=0.2)
        extended = torch.cat((inputs, torch.randn(8, 2)), dim=1)
        torch.testing.assert_close(target(extended), source(inputs))
        torch.testing.assert_close(target.distribution.log_std_param, torch.full((2,), math.log(0.2)))

    def test_checkpoint_loads_both_models_without_training_state(self):
        actor, critic = Actor(3), Actor(3)
        restored_actor, restored_critic = Actor(3), Actor(3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save({"actor_state_dict": actor.state_dict(), "critic_state_dict": critic.state_dict(), "iter": 1499}, path)
            initialize_models(restored_actor, restored_critic, checkpoint_path=path)
        inputs = torch.randn(4, 3)
        torch.testing.assert_close(restored_actor(inputs), actor(inputs))
        torch.testing.assert_close(restored_critic(inputs), critic(inputs))

    def test_invalid_initialization_is_rejected(self):
        with self.assertRaises(ValueError):
            initialize_models(Actor(3), Actor(3))
        with self.assertRaises(ValueError):
            initialize_models(Actor(3), Actor(3), actor_path="unused", initial_std=0.0)


if __name__ == "__main__":
    unittest.main()