"""Single raised box with a ground-level approach origin."""

import numpy as np
import trimesh


def single_box_terrain(difficulty: float, cfg):
    if not 0.0 <= difficulty <= 1.0:
        raise ValueError("Terrain difficulty must be between zero and one")
    height = cfg.box_height_range[0] + difficulty * (cfg.box_height_range[1] - cfg.box_height_range[0])
    if height <= 0.0 or min(cfg.box_length, cfg.box_width, cfg.approach_distance) <= 0.0:
        raise ValueError("Box dimensions and approach distance must be positive")
    center = np.asarray(cfg.size, dtype=np.float64) / 2.0
    approach_height = getattr(cfg, "approach_height", 0.0)
    approach_gap = getattr(cfg, "approach_gap", 0.0)
    gap_range = getattr(cfg, "approach_gap_range", None)
    if gap_range is not None:
        gap_range = np.asarray(gap_range, dtype=np.float64)
        if gap_range.shape != (2,) or not np.isfinite(gap_range).all() or gap_range[0] < 0.0 or gap_range[1] < gap_range[0]:
            raise ValueError("Approach gap range must be finite, ordered and nonnegative")
        approach_gap = gap_range[0] + difficulty * (gap_range[1] - gap_range[0])
    if not np.isfinite(approach_height) or not np.isfinite(approach_gap) or min(approach_height, approach_gap) < 0.0:
        raise ValueError("Approach height and gap must be finite and nonnegative")
    origin = np.array([center[0] - cfg.box_length / 2.0 - cfg.approach_distance, center[1], approach_height])
    if origin[0] < 0.5 or cfg.box_width >= cfg.size[1] or cfg.box_length >= cfg.size[0]:
        raise ValueError("Terrain is too small for the box and ground approach")
    floor = trimesh.creation.box(extents=[cfg.size[0], cfg.size[1], 0.2])
    floor.apply_translation([center[0], center[1], -0.1])
    box = trimesh.creation.box(extents=[cfg.box_length, cfg.box_width, height + approach_height])
    box.apply_translation([center[0], center[1], (height + approach_height) / 2.0])
    meshes = [floor, box]
    if approach_height > 0.0:
        approach_length = cfg.approach_length
        approach_width = cfg.approach_width
        if min(approach_length, approach_width) <= 0.0 or not approach_gap < cfg.approach_distance < approach_gap + approach_length:
            raise ValueError("Spawn must lie inside the raised approach support")
        approach = trimesh.creation.box(extents=[approach_length, approach_width, approach_height])
        approach.apply_translation([center[0] - cfg.box_length / 2.0 - approach_gap - approach_length / 2.0, center[1], approach_height / 2.0])
        meshes.append(approach)
    following_height = getattr(cfg, "following_platform_height", 0.0)
    if following_height > 0.0:
        following_length = cfg.following_platform_length
        following_width = cfg.following_platform_width
        following_gap = cfg.following_platform_gap
        right_edge = center[0] + cfg.box_length / 2.0 + following_gap + following_length
        if following_height <= height + approach_height or following_gap < 0.0 or min(following_length, following_width) <= 0.0 or right_edge > cfg.size[0]:
            raise ValueError("Following platform must fit beyond the target box")
        platform = trimesh.creation.box(extents=[following_length, following_width, following_height])
        platform.apply_translation([right_edge - following_length / 2.0, center[1], following_height / 2.0])
        meshes.append(platform)
    return meshes, origin