"""Run reproducible passage parameter sweeps with physical skill records."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import uuid

import torch

from reconfigurable_navigation.data.snapshot import SimulatorSnapshot
from reconfigurable_navigation.data.snapshot_io import _code_fingerprint, _versions
from reconfigurable_navigation.locomotion_runtime import LocomotionRuntime
from reconfigurable_navigation.passage_scene import ParameterizedPassageEnv, PassageScene, passage_sweep
from reconfigurable_navigation.representations import SkillType
from reconfigurable_navigation.runtime import ReconfigurableExecutor


def run_case(scene: PassageScene, seed: int, case: str, output: Path, *, snapshots: bool, group_id: str) -> dict:
    env = ParameterizedPassageEnv(scene)
    env.reset(seed)
    runtime = LocomotionRuntime(env)
    executor = ReconfigurableExecutor(env, runtime)
    episode_id = uuid.uuid4().hex
    metadata = {
        "episode_id": episode_id, "seed": seed, "scenario": "parameterized_passage",
        "scene_family": "blocked_passage", "scene_id": scene.scene_id,
        "scene_parameters": scene.to_dict(), "sweep_case": case, "sweep_group_id": group_id,
        "capability_profile_id": "go2_arx5_rule_baseline_v1", "capability": asdict(env.capability),
    }
    transitions = []
    snapshot_metadata = {}
    observed = {"illegal_collision": False, "invalid_robot_state": False, "body_contact": False}

    def on_step(_action, _skill, observation):
        observed["illegal_collision"] |= observation.illegal_collision
        observed["invalid_robot_state"] |= not observation.state_valid
        observed["body_contact"] |= bool(observation.body_contact_object_ids)
        return True

    def on_skill_start(_action, previous_skill):
        path = output / "snapshots" / f"{episode_id}-{len(transitions)}.snapshot"
        SimulatorSnapshot.capture(executor, previous_skill).save(path)
        snapshot_metadata.clear()
        snapshot_metadata.update(
            snapshot_file=os.path.relpath(path, output),
            snapshot_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )

    def on_transition(transition):
        line = transition.to_json({**metadata, "skill_index": len(transitions), **snapshot_metadata})
        with (output / "transitions.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        transitions.append(transition)

    result = executor.run(
        on_step=on_step, on_transition=on_transition,
        on_skill_start=on_skill_start if snapshots else None,
    )
    pushes = [transition for transition in transitions if transition.action.skill == SkillType.PUSH]
    push_contact = all(
        any(event.kind == "end_effector_contact" and event.active and event.object_id == transition.action.object_id
            for event in transition.events)
        for transition in pushes
    ) if pushes else None
    accepted = result.succeeded and env.reset_count == 1 and not any(observed.values()) and push_contact is not False
    summary = {
        **metadata, "succeeded": result.succeeded, "accepted": accepted, "reason": result.reason,
        "records": len(transitions), "replans": result.replans, "skill_failures": result.skill_failures,
        "skills": [record.action.skill.name for record in result.records],
        "reset_count": env.reset_count, "elapsed_sim_time": float(env.data.time),
        "process_observed": observed, "push_fingertip_contact": push_contact,
        "final_robot_state": env.observe().robot_state.tolist(),
    }
    with (output / "episodes.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(summary, allow_nan=False) + "\n")
    print(f"CASE seed={seed} case={case} accepted={accepted} reason={result.reason} records={len(transitions)}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True, help="New, non-existing data directory.")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--cases", nargs="+", choices=tuple(passage_sweep(0)))
    parser.add_argument("--scene-json", type=Path, help="One explicit PassageScene parameter object instead of a sweep.")
    parser.add_argument("--snapshots", action="store_true", help="Save trusted skill-start archives for candidate rollout.")
    args = parser.parse_args()
    if args.seeds < 1 or args.seed_offset < 0:
        parser.error("Invalid seed count or offset")
    if args.scene_json is not None and args.cases is not None:
        parser.error("--scene-json and --cases are mutually exclusive")
    if args.cases is not None and len(set(args.cases)) != len(args.cases):
        parser.error("Cases must be unique")
    custom = None if args.scene_json is None else PassageScene(**json.loads(args.scene_json.read_text()))
    torch.set_num_threads(1)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "transitions.jsonl").touch(exist_ok=False)
    (output / "episodes.jsonl").touch(exist_ok=False)
    repository = Path(__file__).resolve().parents[1]
    manifest = {
        "schema_version": 1, "complete": False, "scene_families": ["blocked_passage"],
        "split": "unsplit", "seed_offset": args.seed_offset, "seeds": args.seeds,
        "cases": ["custom"] if custom else args.cases or list(passage_sweep(0)),
        "snapshots": args.snapshots, "runtime_versions": _versions(),
        "code_sha256": _code_fingerprint(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git_commit": subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip(),
        "limitations": ["one passage family, not a held-out dataset", "rule capability profile, not new measured skill limits", "failures do not prove physical infeasibility"],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    results = []
    for seed in range(args.seed_offset, args.seed_offset + args.seeds):
        scenes = {"custom": custom} if custom else passage_sweep(seed)
        group_id = f"passage-{custom.scene_id if custom else scenes['baseline'].scene_id}"
        for case in manifest["cases"]:
            results.append(run_case(scenes[case], seed, case, output, snapshots=args.snapshots, group_id=group_id))
    manifest.update(
        complete=True, episodes=len(results), accepted=sum(result["accepted"] for result in results),
        transition_count=sum(result["records"] for result in results),
        transitions_sha256=hashlib.sha256((output / "transitions.jsonl").read_bytes()).hexdigest(),
        episodes_sha256=hashlib.sha256((output / "episodes.jsonl").read_bytes()).hexdigest(),
    )
    replacement = output / "manifest.complete.json"
    replacement.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    replacement.replace(manifest_path)
    print(f"PASSAGE SWEEP COMPLETE: {manifest['accepted']}/{len(results)} accepted; {output}")


if __name__ == "__main__":
    main()