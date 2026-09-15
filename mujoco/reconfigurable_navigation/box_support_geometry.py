"""Geometric request bounds for the frozen box-support controllers."""

import numpy as np

from .representations import SkillType


def nominal_push_target(box, platform):
    return np.array([platform.center[0] - platform.size[0] / 2.0 - box.size[0] / 2.0 + 0.02, platform.center[1], 0.0])


def push_target_supported(target, box, platform):
    if target.shape != (3,) or not np.isfinite(target).all():
        return False
    if abs(platform.yaw) > 1e-5 or abs(target[2]) > 1e-5:
        return False
    nominal_x = nominal_push_target(box, platform)[0]
    box_half_width = (abs(np.sin(box.yaw)) * box.size[0] + abs(np.cos(box.yaw)) * box.size[1]) / 2.0
    lateral_limit = min(0.08, platform.size[1] / 2.0 - box_half_width - 0.04)
    return bool(
        abs(target[0] - nominal_x) <= 0.04 + 1e-5
        and abs(target[1] - platform.center[1]) <= lateral_limit + 1e-5
        and target[0] > box.center[0] + 0.10
    )


def surface_action_offset(action, surface, capability, standing_surface=None):
    expected = surface.climb_entry_pose if action.skill == SkillType.NAV else surface.climb_landing_pose
    target = action.target_pose
    if action.skill not in (SkillType.NAV, SkillType.CLIMB) or target.shape != expected.shape or not np.isfinite(target).all():
        return None
    rotation = np.array([[np.cos(surface.yaw), -np.sin(surface.yaw)], [np.sin(surface.yaw), np.cos(surface.yaw)]])
    offset = rotation.T @ (target[:2] - expected[:2])
    heading = np.arctan2(np.sin(target[-1] - expected[-1]), np.cos(target[-1] - expected[-1]))
    if action.skill == SkillType.NAV:
        if np.max(np.abs(offset)) > 0.08 + 1e-5 or abs(heading) > 0.08 + 1e-5:
            return None
        if standing_surface is not None:
            standing_rotation = np.array([[np.cos(standing_surface.yaw), -np.sin(standing_surface.yaw)], [np.sin(standing_surface.yaw), np.cos(standing_surface.yaw)]])
            relative = standing_rotation.T @ (target[:2] - standing_surface.center[:2])
            margin = standing_surface.size[:2] / 2.0 - np.array([capability.body_length, capability.body_width]) / 2.0 - 0.04
            if np.any(np.abs(relative) > margin):
                return None
        result = np.array([*offset, heading])
    else:
        limit = np.minimum([0.10, 0.08], surface.size[:2] / 2.0 - np.array([capability.body_length, capability.body_width]) / 2.0 - 0.04)
        if abs(target[2] - expected[2]) > 1e-5 or abs(heading) > 1e-5 or np.any(np.abs(offset) > limit + 1e-5):
            return None
        result = offset
    if np.allclose(target, expected, rtol=0.0, atol=1e-5):
        result.fill(0.0)
    return result