"""Completion credit shared by box skills."""

import torch


def support_clearance(base_height, foot_heights, normal_forces):
    supported = normal_forces > 2.0
    surfaces = torch.where(supported, foot_heights - 0.0225, torch.inf)
    lowest = surfaces.min(dim=1).values
    lowest = torch.where(torch.isfinite(lowest), lowest, torch.zeros_like(lowest))
    return base_height - lowest


def velocity_tracking_advantage(command, velocity, std=0.4):
    error = torch.sum((command - velocity).square(), dim=1)
    stationary_error = torch.sum(command.square(), dim=1)
    return torch.exp(-error / std**2) - torch.exp(-stationary_error / std**2)


def top_surface_support(foot_heights, normal_forces, surface_height):
    height = surface_height[:, None]
    return (
        torch.isfinite(height)
        & (foot_heights >= height - 0.01)
        & (foot_heights <= height + 0.06)
        & (normal_forces > 2.0)
    )


def directed_progress(position, target, velocity, valid_contact, speed_scale=0.3):
    delta = target - position
    direction = delta / torch.linalg.vector_norm(delta, dim=1, keepdim=True).clamp_min(1.0e-6)
    toward_target = torch.sum(velocity * direction, dim=1)
    return torch.clamp(toward_target / speed_scale, -1.0, 1.0) * valid_contact.float()


def valid_skill_success(env, failure_terms, success_term="box_settled"):
    success = env.termination_manager.get_term(success_term).clone()
    for name in failure_terms:
        success &= ~env.termination_manager.get_term(name)
    return success


def completion_reward(env, failure_terms, success_term="box_settled"):
    if env.step_dt <= 0.0:
        raise ValueError("step_dt must be positive")
    return valid_skill_success(env, failure_terms, success_term).float() / env.step_dt


def failure_cost(env, failure_terms):
    failures = torch.stack([env.termination_manager.get_term(name) for name in failure_terms], dim=1).any(dim=1)
    return failures.float() / env.step_dt