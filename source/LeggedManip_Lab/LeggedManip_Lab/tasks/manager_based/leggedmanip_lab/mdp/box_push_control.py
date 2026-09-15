"""Contact-aware locomotion commands for physical pushing."""

import torch


def coordinated_joint_target(reference, desired, max_delta):
    delta = desired - reference
    scale = torch.clamp(max_delta / delta.abs().amax(dim=1, keepdim=True).clamp_min(1.0e-9), max=1.0)
    return reference + scale * delta


def side_approach_ready(hand_local, box_half_length, approach_distance):
    return (
        (hand_local[:, 0] >= -box_half_length - approach_distance - 0.06)
        & (hand_local[:, 0] <= -box_half_length - 0.03)
        & (hand_local[:, 1].abs() < 0.10)
        & (hand_local[:, 2].abs() < 0.035)
    )


def valid_front_contact(force, box_axis, contact_local, near_tip, box_size):
    strength = torch.linalg.vector_norm(force, dim=1)
    resisting_force = -torch.sum(force * box_axis, dim=1)
    return (
        near_tip & torch.isfinite(contact_local).all(dim=1)
        & (resisting_force > 1.0) & (resisting_force > 0.8 * strength)
        & ((contact_local[:, 0] + box_size[0] / 2.0).abs() < 0.02)
        & (contact_local[:, 1].abs() < box_size[1] / 2.0)
        & (contact_local[:, 2].abs() < box_size[2] / 2.0 - 0.01)
    )


def contact_push_velocity(face, box_velocity, target_distance, contact, heading_error=None, drive_speed=0.18, min_gap=0.43):
    commands = torch.zeros_like(face)
    aligned = face[:, 1].abs() < 0.12
    if heading_error is not None:
        aligned &= heading_error.abs() < 0.25
        commands[:, 2] = torch.clamp(1.5 * heading_error, -0.4, 0.4)
    drive = (contact & aligned).float() * drive_speed
    commands[:, 0] = torch.clamp(box_velocity[:, 0] + 0.8 * (face[:, 0] - 0.50) + drive, -0.10, 0.25)
    commands[:, 0] = torch.where(commands[:, 0] > 0.01, commands[:, 0].clamp_min(0.15), commands[:, 0])
    commands[:, 0] = torch.where(face[:, 0] <= min_gap, commands[:, 0].clamp_max(0.0), commands[:, 0])
    commands[:, 1] = torch.clamp(0.8 * face[:, 1], -0.12, 0.12)
    commands[target_distance < 0.10] = 0.0
    return commands