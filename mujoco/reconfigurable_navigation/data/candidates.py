"""Local action perturbations for physical data collection, not a planner."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json

import numpy as np

from ..representations import SkillAction, SkillType
from ..box_support_scene import BOX_SUPPORT_FAMILIES
from .events import EVENT_KINDS
from .snapshot import SkillReplay


@dataclass(frozen=True)
class SkillCandidate:
    source: str
    action: SkillAction


def candidate_scene_metadata(record: dict) -> dict:
    metadata = record["metadata"]
    family = metadata.get("scene_family")
    if family is None and metadata.get("scenario") == "complex_course":
        family = "complex_course_fixed_layout"
    if not isinstance(family, str) or not family:
        raise ValueError("Source record must identify its scene family")
    fields = ("scene_id", "scene_parameters", "scene_case", "dataset_split", "split_group_id", "sweep_case", "sweep_group_id", "capability_profile_id", "capability")
    return {"scene_family": family, **{name: deepcopy(metadata[name]) for name in fields if name in metadata}}


def candidate_partition_metadata(records):
    assignments = {}
    grouped = [record["metadata"].get("dataset_split") is not None for record in records]
    if not any(grouped):
        return {"split": "pilot_unsplit", "family_splits": {}}
    if not all(grouped):
        raise ValueError("Cannot mix grouped and unpartitioned source records")
    group_splits = {}
    for record in records:
        metadata = candidate_scene_metadata(record)
        split = metadata["dataset_split"]
        if split not in ("train", "validation", "test"):
            raise ValueError("Unsupported dataset split")
        family = metadata["scene_family"]
        group = metadata.get("split_group_id", family)
        if assignments.setdefault(family, split) != split or group_splits.setdefault(group, split) != split:
            raise ValueError("A scene family or source group crosses dataset splits")
    return {"split": "scene_family_v1", "family_splits": assignments}


def build_candidates(action: SkillAction, *, count: int, seed: int, observation=None, scene_family: str | None = None) -> tuple[SkillCandidate, ...]:
    if count < 3:
        raise ValueError("At least three candidates are required")
    expected_size = 4 if action.skill == SkillType.CLIMB else 3
    if action.skill not in (SkillType.NAV, SkillType.PUSH, SkillType.CLIMB):
        raise ValueError("Candidate collection only supports executable skills")
    if action.target_pose.shape != (expected_size,) or not np.isfinite(action.target_pose).all():
        raise ValueError("Reference action must be finite with the expected target shape")
    if scene_family in BOX_SUPPORT_FAMILIES:
        if observation is None:
            raise ValueError("Box-support candidates require snapshot observations")
        return _box_support_candidates(action, observation, count=count, seed=seed)
    generator = np.random.default_rng(seed)
    candidates = [SkillCandidate("reference", deepcopy(action))]
    for index in range(count - 2):
        boundary = index % 2 == 1
        target = action.target_pose.copy()
        radius = 0.5 if boundary else 0.12
        target[:2] += generator.uniform(-radius, radius, size=2)
        yaw_radius = 0.6 if boundary else 0.12
        target[-1] += generator.uniform(-yaw_radius, yaw_radius)
        target[-1] = (target[-1] + np.pi) % (2.0 * np.pi) - np.pi
        if action.skill == SkillType.CLIMB:
            target[2] += generator.uniform(-0.05, 0.2) if boundary else 0.0
        candidates.append(
            SkillCandidate(
                "boundary_parameter" if boundary else "nearby_parameter",
                SkillAction(action.skill, target, action.object_id, action.support_id),
            )
        )
    invalid = (
        SkillAction(action.skill, action.target_pose.copy(), object_id=-1, support_id=action.support_id)
        if action.skill == SkillType.PUSH
        else SkillAction(action.skill, np.empty(0), action.object_id, action.support_id)
    )
    candidates.append(SkillCandidate("invalid_request", invalid))
    return tuple(candidates)


def _box_support_candidates(action, observation, *, count, seed):
    from ..box_support_env import BOX_ID, PLATFORM_ID
    from ..box_support_geometry import nominal_push_target, push_target_supported, surface_action_offset

    objects = {obj.object_id: obj for obj in observation.objects}
    box, platform = objects[BOX_ID], objects[PLATFORM_ID]
    surface = None if action.skill == SkillType.PUSH else objects[action.support_id]
    if action.skill == SkillType.PUSH:
        nominal = nominal_push_target(box, platform)
        rotation = np.eye(2)
        scale = np.array([0.04, 0.06])
        source = "support_parking"
    else:
        nominal = surface.climb_entry_pose if action.skill == SkillType.NAV else surface.climb_landing_pose
        rotation = np.array([[np.cos(surface.yaw), -np.sin(surface.yaw)], [np.sin(surface.yaw), np.cos(surface.yaw)]])
        scale = np.array([0.06, 0.06]) if action.skill == SkillType.NAV else np.array([0.08, 0.06])
        source = "support_approach" if action.skill == SkillType.NAV else "support_landing"
    generator = np.random.default_rng(seed)
    offsets = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0], [-0.75, 0.75], [0.75, -0.75]])
    generator.shuffle(offsets)
    candidates = [SkillCandidate("reference", deepcopy(action))]
    seen = {tuple(action.target_pose.tolist())}
    for attempt in range(max(128, count * 20)):
        offset = offsets[attempt] if attempt < len(offsets) else generator.uniform(-1.0, 1.0, 2)
        target = nominal.copy()
        target[:2] += rotation @ (offset * scale)
        proposed = SkillAction(action.skill, target, action.object_id, action.support_id)
        if action.skill == SkillType.PUSH:
            valid = push_target_supported(proposed.target_pose, box, platform)
        else:
            standing = box if action.support_id == PLATFORM_ID else None
            valid = surface_action_offset(proposed, surface, observation.capability, standing) is not None
        key = tuple(proposed.target_pose.tolist())
        if valid and key not in seen:
            candidates.append(SkillCandidate(source, proposed))
            seen.add(key)
        if len(candidates) == count - 1:
            break
    if len(candidates) != count - 1:
        raise ValueError("Insufficient candidates inside the supported geometric region")
    invalid = SkillAction(action.skill, np.empty(0), action.object_id, action.support_id)
    candidates.append(SkillCandidate("invalid_request", invalid))
    return tuple(candidates)


def candidate_payload(candidate: SkillCandidate, result: SkillReplay) -> dict:
    transition = None if result.transition is None else json.loads(result.transition.to_json())
    executed = transition is not None
    truncated = result.reason == "rollout_budget_exhausted"
    outcome = "rejected" if not executed else "truncated" if truncated else transition["outcome"]
    process_labels = dict.fromkeys(EVENT_KINDS) if transition is None else transition["process_labels"]
    return {
        "schema_version": 2,
        "candidate_source": candidate.source,
        "action": {
            "skill": int(candidate.action.skill),
            "target_pose": candidate.action.target_pose.tolist(),
            "object_id": candidate.action.object_id,
            "support_id": candidate.action.support_id,
        },
        "executed": executed,
        "truncated": truncated,
        "outcome": outcome,
        "reason": result.reason if transition is None else transition["termination_reason"],
        "failure_reason": None if transition is None else transition["failure_reason"],
        "skill_success": None if transition is None else transition["skill_success"],
        "process_labels": process_labels,
        "label_validity": {
            "dynamics": executed,
            "skill_success": executed and not transition["interrupted"],
            "reachability": False,
            "support_stable": False,
            "failure_reason": executed and transition["failure_reason"] is not None,
            **{
                kind: value is not None and (not transition["interrupted"] or value)
                for kind, value in process_labels.items()
            },
        },
        "transition": transition,
    }