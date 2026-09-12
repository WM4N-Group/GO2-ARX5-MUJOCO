"""Local action perturbations for physical data collection, not a planner."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json

import numpy as np

from ..representations import SkillAction, SkillType
from .events import EVENT_KINDS
from .snapshot import SkillReplay


@dataclass(frozen=True)
class SkillCandidate:
    source: str
    action: SkillAction


def build_candidates(action: SkillAction, *, count: int, seed: int) -> tuple[SkillCandidate, ...]:
    if count < 3:
        raise ValueError("At least three candidates are required")
    expected_size = 4 if action.skill == SkillType.CLIMB else 3
    if action.skill not in (SkillType.NAV, SkillType.PUSH, SkillType.CLIMB):
        raise ValueError("Candidate collection only supports executable skills")
    if action.target_pose.shape != (expected_size,) or not np.isfinite(action.target_pose).all():
        raise ValueError("Reference action must be finite with the expected target shape")
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