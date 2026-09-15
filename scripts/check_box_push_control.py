"""Check contact drive, separation limits and goal stopping."""

from pathlib import Path
import runpy
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/mdp/box_push_control.py"
helpers = runpy.run_path(str(MODULE))
command = helpers["contact_push_velocity"]


class PushControlChecks(unittest.TestCase):
    def test_joint_rate_limit_preserves_coordinated_direction(self):
        reference = torch.zeros((2, 2))
        desired = torch.tensor([[1.0, 0.1], [0.05, -0.02]])
        actual = helpers["coordinated_joint_target"](reference, desired, 0.1)
        torch.testing.assert_close(actual, torch.tensor([[0.1, 0.01], [0.05, -0.02]]))
        torch.testing.assert_close(reference, torch.zeros_like(reference))

    def test_approach_must_be_outside_face_at_contact_height(self):
        hands = torch.tensor([[-0.7, 0.0, 0.0], [-0.7, 0.0, 0.15], [-0.5, 0.0, 0.0], [-0.7, 0.2, 0.0], [-1.0, 0.0, 0.0]])
        ready = helpers["side_approach_ready"](hands, 0.6, 0.10)
        torch.testing.assert_close(ready, torch.tensor([True, False, False, False, False]))

    def test_top_rear_and_wrist_contacts_are_not_push_contacts(self):
        forces = torch.tensor([[-10.0, 0.0, 0.0], [0.0, 0.0, 10.0], [10.0, 0.0, 0.0], [-10.0, 0.0, 0.0], [-10.0, 0.0, 0.0]])
        points = torch.tensor([[-0.6, 0.0, 0.0]]).expand(5, -1).clone()
        points[4, 2] = 0.125
        axes = torch.tensor([[1.0, 0.0, 0.0]]).expand(5, -1)
        contact = helpers["valid_front_contact"](forces, axes, points, torch.tensor([True, True, True, False, True]), (1.2, 1.2, 0.25))
        torch.testing.assert_close(contact, torch.tensor([True, False, False, False, False]))

    def test_heading_feedback_turns_toward_push_direction_before_driving(self):
        faces = torch.tensor([[0.5, 0.0, 0.0]]).expand(3, -1)
        velocity = command(faces, torch.zeros_like(faces), torch.ones(3), torch.ones(3, dtype=torch.bool), torch.tensor([0.6, -0.6, 0.1]))
        torch.testing.assert_close(velocity[:, 2], torch.tensor([0.4, -0.4, 0.15]))
        torch.testing.assert_close(velocity[:, 0], torch.tensor([0.0, 0.0, 0.18]))

    def test_static_box_receives_drive_only_after_aligned_contact(self):
        faces = torch.tensor([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.2, 0.0]])
        velocity = command(faces, torch.zeros_like(faces), torch.ones(3), torch.tensor([False, True, True]))
        torch.testing.assert_close(velocity[:, 0], torch.tensor([0.0, 0.18, 0.0]))

    def test_approach_exceeds_policy_deadzone_without_forcing_small_errors(self):
        faces = torch.tensor([[0.55, 0.0, 0.0], [0.51, 0.0, 0.0]])
        velocity = command(faces, torch.zeros_like(faces), torch.ones(2), torch.zeros(2, dtype=torch.bool))
        torch.testing.assert_close(velocity[:, 0], torch.tensor([0.15, 0.008]))

    def test_contact_drive_never_overrides_separation_or_goal_stop(self):
        faces = torch.tensor([[0.40, 0.0, 0.0], [0.43, 0.0, 0.0], [0.5, 0.0, 0.0]])
        boxes = torch.tensor([[0.2, 0.0, 0.0]]).expand(3, -1)
        velocity = command(faces, boxes, torch.tensor([1.0, 1.0, 0.05]), torch.ones(3, dtype=torch.bool))
        torch.testing.assert_close(velocity, torch.zeros_like(faces))

    def test_far_approach_stays_bounded_and_input_is_not_mutated(self):
        face = torch.tensor([[1.4, -0.5, 0.0]])
        original = face.clone()
        velocity = command(face, torch.zeros_like(face), torch.ones(1), torch.zeros(1, dtype=torch.bool))
        torch.testing.assert_close(velocity, torch.tensor([[0.25, -0.12, 0.0]]))
        torch.testing.assert_close(face, original)

    def test_crossed_goal_stops_despite_lateral_error_and_box_motion(self):
        face = torch.tensor([[0.6, 0.2, 0.0]])
        velocity = command(face, torch.tensor([[0.5, 0.0, 0.0]]), torch.tensor([-0.05]), torch.tensor([True]), torch.tensor([2.0]))
        torch.testing.assert_close(velocity, torch.zeros_like(face))


if __name__ == "__main__":
    unittest.main()