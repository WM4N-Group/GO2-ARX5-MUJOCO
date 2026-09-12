"""Replay trusted skill snapshots and compare them with recorded observations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from reconfigurable_navigation.data.snapshot import SimulatorSnapshot
from reconfigurable_navigation.representations import SkillAction, SkillType


def compare_record(actual: object, expected: object, atol: float, path: str = "record") -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise AssertionError(f"Different fields at {path}")
        for key, value in expected.items():
            compare_record(actual[key], value, atol, f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError(f"Different list shape at {path}")
        for index, (result, value) in enumerate(zip(actual, expected)):
            compare_record(result, value, atol, f"{path}[{index}]")
    elif isinstance(expected, float):
        if not isinstance(actual, (int, float)) or not math.isclose(actual, expected, rel_tol=0.0, abs_tol=atol):
            raise AssertionError(f"Numeric mismatch at {path}: {actual} != {expected}")
    elif actual != expected:
        raise AssertionError(f"Mismatch at {path}: {actual} != {expected}")


def replay_record(record: dict, source: Path, repeats: int = 3, atol: float = 1.0e-6) -> None:
    if record["schema_version"] not in (1, 2):
        raise ValueError("Unsupported transition schema")
    metadata = record["metadata"]
    snapshot_path = source.parent / metadata["snapshot_file"]
    if hashlib.sha256(snapshot_path.read_bytes()).hexdigest() != metadata["snapshot_sha256"]:
        raise ValueError("Snapshot file checksum does not match the recorded transition")
    snapshot = SimulatorSnapshot.load(snapshot_path, trusted=True)
    action = SkillAction(
        skill=SkillType(record["action"]["skill"]),
        target_pose=np.asarray(record["action"]["target_pose"]),
        object_id=record["action"]["object_id"],
        support_id=record["action"]["support_id"],
    )
    limit = None
    if record["interrupted"]:
        limit = max(1, round(record["elapsed_sim_time"] / record["control_dt"]))
    expected = {key: value for key, value in record.items() if key != "metadata"}
    for _ in range(repeats):
        result = snapshot.rollout(action, max_control_steps=limit)
        if result.transition is None:
            raise AssertionError(f"Recorded action rejected during replay: {result.reason}")
        actual = json.loads(result.transition.to_json())
        actual.pop("metadata")
        comparison = expected.copy()
        if record["schema_version"] == 1:
            actual = {key: actual[key] for key in expected}
            actual["schema_version"] = 1
        elif record["interrupted"]:
            if actual.pop("termination_reason") != "rollout_budget_exhausted":
                raise AssertionError("Interrupted replay must stop at the recorded budget")
            comparison.pop("termination_reason")
        compare_record(actual, comparison, atol)
    print(
        f"REPLAY PASSED episode={metadata['episode_id']} skill_index={metadata['skill_index']} "
        f"skill={action.skill.name} outcome={record['outcome']} repeats={repeats} atol={atol}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-jsonl", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="+", default=[0], help="Zero-based JSONL record indices.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--atol", type=float, default=1.0e-6)
    parser.add_argument("--trusted", action="store_true", help="Trust the native state and TorchScript in these snapshots.")
    args = parser.parse_args()
    if not args.trusted:
        parser.error("Only load self-generated, trusted snapshots; --trusted is required")
    if args.repeats < 1 or not math.isfinite(args.atol) or args.atol < 0:
        parser.error("Invalid replay count or tolerance")
    if any(index < 0 for index in args.indices):
        parser.error("Record indices must be non-negative")
    selected = set(args.indices)
    with args.record_jsonl.open(encoding="utf-8") as records:
        for index, line in enumerate(records):
            if index in selected:
                replay_record(json.loads(line), args.record_jsonl, args.repeats, args.atol)
                selected.remove(index)
            if not selected:
                break
    if selected:
        parser.error(f"Record indices do not exist: {sorted(selected)}")


if __name__ == "__main__":
    main()