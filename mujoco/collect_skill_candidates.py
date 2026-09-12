"""Collect physical counterfactual skill outcomes from trusted start snapshots."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import time
from zipfile import ZipFile

import numpy as np
import torch

from reconfigurable_navigation.data.candidates import build_candidates, candidate_payload
from reconfigurable_navigation.data.snapshot import SimulatorSnapshot
from reconfigurable_navigation.representations import SkillAction, SkillType


def collect_snapshot(job: dict) -> list[dict]:
    torch.set_num_threads(1)
    source_record = job["record"]
    source_path = Path(job["source_path"])
    snapshot_path = source_path.parent / source_record["metadata"]["snapshot_file"]
    digest = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    if digest != source_record["metadata"]["snapshot_sha256"]:
        raise ValueError("Source snapshot checksum mismatch")
    snapshot = SimulatorSnapshot.load(snapshot_path, trusted=True)
    with ZipFile(snapshot_path) as archive:
        snapshot_manifest = json.loads(archive.read("manifest.json"))
    action = source_record["action"]
    reference = SkillAction(
        SkillType(action["skill"]), np.asarray(action["target_pose"]),
        action["object_id"], action["support_id"],
    )
    candidate_seed = int(np.random.SeedSequence([job["seed"], job["record_index"]]).generate_state(1)[0])
    output = []
    for candidate_index, candidate in enumerate(build_candidates(reference, count=job["count"], seed=candidate_seed)):
        started = time.perf_counter()
        result = snapshot.rollout(candidate.action, max_control_steps=job["max_control_steps"])
        payload = candidate_payload(candidate, result)
        identifier = f"{job['source_sha256']}:{job['record_index']}:{candidate_seed}:{candidate_index}"
        payload.update(
            candidate_id=hashlib.sha256(identifier.encode()).hexdigest(),
            source_record_index=job["record_index"],
            episode_id=source_record["metadata"]["episode_id"],
            scene_seed=source_record["metadata"]["seed"],
            scene_family="complex_course_fixed_layout",
            candidate_index=candidate_index,
            candidate_seed=candidate_seed,
            max_control_steps=job["max_control_steps"],
            snapshot_file=os.path.relpath(snapshot_path.resolve(), job["output_dir"]),
            snapshot_sha256=digest,
            snapshot_code_sha256=snapshot_manifest["code_sha256"],
            runtime_versions=snapshot_manifest["versions"],
            wall_seconds=time.perf_counter() - started,
        )
        output.append(payload)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New, non-existing dataset directory.")
    parser.add_argument("--indices", type=int, nargs="+", help="Optional zero-based source record indices.")
    parser.add_argument("--candidates-per-snapshot", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-control-steps", type=int, default=5000)
    parser.add_argument("--trusted", action="store_true")
    args = parser.parse_args()
    if not args.trusted:
        parser.error("Only use self-generated trusted snapshots; --trusted is required")
    if args.candidates_per_snapshot < 3 or args.seed < 0 or args.workers < 1 or args.max_control_steps < 1:
        parser.error("Invalid candidate count, seed, worker count, or control budget")
    if args.indices is not None and (min(args.indices) < 0 or len(set(args.indices)) != len(args.indices)):
        parser.error("Record indices must be unique and non-negative")
    source_path = args.record_jsonl.resolve()
    raw_source = source_path.read_bytes()
    records = [json.loads(line) for line in raw_source.splitlines()]
    indices = list(range(len(records))) if args.indices is None else sorted(args.indices)
    if not indices or max(indices) >= len(records):
        parser.error("Source record indices do not exist")
    for index in indices:
        record = records[index]
        if record["schema_version"] != 1 or "snapshot_file" not in record["metadata"]:
            parser.error("Every selected record must reference a compatible snapshot")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    source_sha256 = hashlib.sha256(raw_source).hexdigest()
    manifest = {
        "schema_version": 1,
        "complete": False,
        "source_jsonl": os.path.relpath(source_path, output_dir),
        "source_sha256": source_sha256,
        "source_indices": indices,
        "seed": args.seed,
        "candidates_per_snapshot": args.candidates_per_snapshot,
        "max_control_steps": args.max_control_steps,
        "workers": args.workers,
        "torch_threads_per_worker": 1,
        "scene_families": ["complex_course_fixed_layout"],
        "split": "pilot_unsplit",
        "limitations": ["local parameter perturbations, not grounded plans", "no reachability or support-stability labels", "single scene family, not held-out evaluation"],
    }
    repository = Path(__file__).resolve().parents[1]
    manifest["git_commit"] = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    manifest["collector_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest["candidate_generator_sha256"] = hashlib.sha256((repository / "mujoco/reconfigurable_navigation/data/candidates.py").read_bytes()).hexdigest()
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    jobs = [
        {
            "record": records[index], "record_index": index,
            "source_path": str(source_path), "source_sha256": source_sha256,
            "output_dir": str(output_dir), "count": args.candidates_per_snapshot,
            "seed": args.seed, "max_control_steps": args.max_control_steps,
        }
        for index in indices
    ]
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    totals = Counter()
    started = time.perf_counter()
    partial = output_dir / "candidates.partial.jsonl"
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        with partial.open("x", encoding="utf-8") as output:
            for source_index, payloads in zip(indices, pool.map(collect_snapshot, jobs)):
                for payload in payloads:
                    output.write(json.dumps(payload, allow_nan=False) + "\n")
                    totals[payload["outcome"]] += 1
                output.flush()
                print(f"COLLECTED source_index={source_index} count={len(payloads)} totals={dict(totals)}", flush=True)
    completed = output_dir / "candidates.jsonl"
    partial.rename(completed)
    manifest.update(
        complete=True,
        records=sum(totals.values()),
        outcomes=dict(totals),
        executed=sum(count for outcome, count in totals.items() if outcome != "rejected"),
        elapsed_wall_seconds=time.perf_counter() - started,
        candidates_sha256=hashlib.sha256(completed.read_bytes()).hexdigest(),
    )
    replacement = output_dir / "manifest.complete.json"
    replacement.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    replacement.replace(manifest_path)
    print(f"CANDIDATE DATASET COMPLETE: {manifest['records']} records; {dict(totals)}; {output_dir}")


if __name__ == "__main__":
    main()