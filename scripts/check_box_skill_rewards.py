"""Check success credit and invalid-completion masking without a simulator."""

from pathlib import Path
import runpy
from types import SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/mdp/box_rewards.py"
helpers = runpy.run_path(str(MODULE))


class BoxRewardChecks(unittest.TestCase):
    def test_simultaneous_failures_have_one_event_cost(self):
        terms = {"collision": torch.tensor([True, False, True]), "posture": torch.tensor([True, False, False])}
        env = SimpleNamespace(step_dt=0.02, termination_manager=SimpleNamespace(get_term=terms.__getitem__))
        result = helpers["failure_cost"](env, ("collision", "posture")) * env.step_dt
        torch.testing.assert_close(result, torch.tensor([1.0, 0.0, 1.0]))

    def test_velocity_tracking_has_no_stationary_reward(self):
        commands = torch.tensor([[0.3, 0.0]]).expand(3, -1)
        velocity = torch.tensor([[0.0, 0.0], [0.3, 0.0], [-0.3, 0.0]])
        result = helpers["velocity_tracking_advantage"](commands, velocity)
        self.assertEqual(result[0].item(), 0.0)
        self.assertGreater(result[1].item(), 0.0)
        self.assertLess(result[2].item(), 0.0)

    def test_clearance_distinguishes_prone_standing_and_raised_support(self):
        base = torch.tensor([0.11, 0.27, 0.52, 0.40])
        feet = torch.tensor([[0.0225] * 4, [0.0225] * 4, [0.2725] * 4, [0.2725, 0.2725, 0.0225, 0.0225]])
        result = helpers["support_clearance"](base, feet, torch.ones_like(feet) * 10.0)
        torch.testing.assert_close(result, torch.tensor([0.11, 0.27, 0.27, 0.40]))

    def test_box_support_rejects_floor_air_and_unknown_surface(self):
        heights = torch.tensor([[0.2725, 0.0225, 0.29, 0.50], [0.2725] * 4])
        forces = torch.tensor([[10.0, 10.0, 0.0, 10.0], [10.0] * 4])
        surface = torch.tensor([0.25, float("nan")])
        result = helpers["top_surface_support"](heights, forces, surface)
        torch.testing.assert_close(result, torch.tensor([[True, False, False, False], [False] * 4]))

    def test_progress_requires_contact_and_correct_direction(self):
        positions = torch.tensor([[0.0, 0.0], [2.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
        targets = torch.tensor([[1.0, 0.0]]).expand(4, -1)
        velocities = torch.tensor([[0.3, 0.0]]).expand(4, -1)
        contact = torch.tensor([True, True, True, False])
        result = helpers["directed_progress"](positions, targets, velocities, contact)
        torch.testing.assert_close(result, torch.tensor([1.0, -1.0, 0.0, 0.0]))

    def test_completion_bonus_is_independent_of_control_period(self):
        for period in (0.01, 0.02, 0.04):
            terms = {"box_settled": torch.tensor([True, False]), "base_contact": torch.tensor([False, False])}
            env = SimpleNamespace(step_dt=period, termination_manager=SimpleNamespace(get_term=terms.__getitem__))
            reward = helpers["completion_reward"](env, ("base_contact",)) * period * 50.0
            torch.testing.assert_close(reward, torch.tensor([50.0, 0.0]))

    def test_unsafe_completion_does_not_receive_success_credit(self):
        terms = {
            "box_settled": torch.tensor([True, True, True, False]),
            "base_contact": torch.tensor([False, True, False, False]),
            "box_tipped": torch.tensor([False, False, True, False]),
        }
        original = terms["box_settled"].clone()
        env = SimpleNamespace(step_dt=0.02, termination_manager=SimpleNamespace(get_term=terms.__getitem__))
        result = helpers["valid_skill_success"](env, ("base_contact", "box_tipped"))
        torch.testing.assert_close(result, torch.tensor([True, False, False, False]))
        torch.testing.assert_close(terms["box_settled"], original)


if __name__ == "__main__":
    unittest.main()