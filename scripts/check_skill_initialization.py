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