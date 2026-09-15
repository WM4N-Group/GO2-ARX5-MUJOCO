"""Run frozen box skills through the project executor and N1 transition recorder."""

import argparse
import hashlib
from itertools import product
import json
import os
from pathlib import Path

import torch

from reconfigurable_navigation.box_robot_profile import box_runtime_provenance
from reconfigurable_navigation.box_support_scene import BoxSupportScene, box_support_suite
from reconfigurable_navigation.data.snapshot import SimulatorSnapshot
from reconfigurable_navigation.runtime.box_support_backend import make_box_support_executor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--push-policy", type=Path, required=True)
    parser.add_argument("--climb-policy", type=Path, required=True)
    parser.add_argument("--platform-policy", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--seed-offset", type=int, default=500)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--record-jsonl", type=Path)
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--snapshot-first-only", action="store_true")
    parser.add_argument("--reference-json", type=Path)
    scenes_group = parser.add_mutually_exclusive_group()
    scenes_group.add_argument("--scene-json", type=Path)
    scenes_group.add_argument("--scene-suite", action="store_true")
    args = parser.parse_args()
    if args.seeds < 1 or args.seed_offset < 0:
        parser.error("Use positive seed count and nonnegative seed offset")
    if args.snapshot_dir is not None and args.record_jsonl is None:
        parser.error("--snapshot-dir requires --record-jsonl")
    if args.snapshot_first_only and args.snapshot_dir is None:
        parser.error("--snapshot-first-only requires --snapshot-dir")
    if args.reference_json is not None and (args.scene_json is not None or args.scene_suite):
        parser.error("The frozen reference is only valid for the default scene")
    scenes = {"nominal": None}
    if args.scene_suite:
        scenes = box_support_suite()
    elif args.scene_json is not None:
        scenes = {"configured": BoxSupportScene(**json.loads(args.scene_json.read_text()))}
    for path in (args.output_json, args.record_jsonl):
        if path is not None and path.exists():
            parser.error("Output files must not already exist")
    torch.set_num_threads(1)
    policies = {role: hashlib.sha256(path.read_bytes()).hexdigest() for role, path in (("push", args.push_policy), ("climb", args.climb_policy), ("platform", args.platform_policy))}
    references = None
    if args.reference_json is not None:
        reference = json.loads(args.reference_json.read_text())
        if reference["policy_sha256"] != policies:
            parser.error("Reference policies differ from selected policies")
        references = {row["seed"]: row for row in reference["results"]}
        if any(seed not in references for seed in range(args.seed_offset, args.seed_offset + args.seeds)):
            parser.error("Reference does not cover all requested seeds")
    provenance = box_runtime_provenance()
    if args.snapshot_dir is not None:
        args.snapshot_dir.mkdir(parents=True, exist_ok=False)
    writer = None
    results = []
    transition_count = 0
    try:
        if args.record_jsonl is not None:
            args.record_jsonl.parent.mkdir(parents=True, exist_ok=True)
            writer = args.record_jsonl.open("x", encoding="utf-8")
        for (case_name, scene), seed in product(scenes.items(), range(args.seed_offset, args.seed_offset + args.seeds)):
            executor = make_box_support_executor(args.push_policy, args.climb_policy, args.platform_policy, seed, scene=scene)
            episode_id = f"box-support-{seed}" if scene is None else f"box-support-{executor.env.scene.scene_id}-{seed}"
            backend = executor.skill_backend
            transitions = []
            starts = []
            primitive_offset = 0
            snapshot_metadata = {}

            def skill_start(action, previous):
                starts.append({"skill": action.skill.name, "sim_time": float(executor.env.data.time), "completed_operations": sorted((kind.name, object_id) for kind, object_id in backend.completed)})
                snapshot_metadata.clear()
                if args.snapshot_dir is not None and (not args.snapshot_first_only or len(starts) == 1):
                    path = args.snapshot_dir / f"{episode_id}-{len(starts) - 1}.snapshot"
                    SimulatorSnapshot.capture(executor, previous).save(path)
                    snapshot_metadata.update(
                        snapshot_file=os.path.relpath(path.resolve(), args.record_jsonl.parent.resolve()),
                        snapshot_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    )

            def record_transition(transition):
                nonlocal primitive_offset, transition_count
                transitions.append(transition)
                transition_count += 1
                if writer is not None:
                    metadata = {
                        "episode_id": episode_id, "scene_seed": seed,
                        "seed": seed, "skill_index": len(starts) - 1,
                        "scene_family": executor.env.scene_family,
                        "scene_id": executor.env.scene.scene_id,
                        "scene_parameters": executor.env.scene.to_dict(),
                        "scene_case": case_name,
                        "split": executor.env.dataset_split,
                        "dataset_split": executor.env.dataset_split if scene is not None else None,
                        "split_group_id": executor.env.scene_family,
                        "driver": "reconfigurable_executor_box_support",
                        "policy_sha256": policies, "runtime_provenance": provenance,
                        "full_snapshot_available": bool(snapshot_metadata),
                        "physical_initializations": executor.env.reset_count,
                        "start_context": starts[-1],
                        "primitive_reports": backend.primitive_reports[primitive_offset:],
                        **snapshot_metadata,
                    }
                    writer.write(transition.to_json(metadata) + "\n")
                    writer.flush()
                primitive_offset = len(backend.primitive_reports)

            outcome = executor.run(on_transition=record_transition, on_skill_start=skill_start)
            stages = backend.primitive_reports
            result = {
                "seed": seed, "succeeded": outcome.succeeded, "reason": outcome.reason,
                "scene_case": case_name, "scene_id": executor.env.scene.scene_id,
                "scene_family": executor.env.scene_family, "dataset_split": executor.env.dataset_split,
                "scene_parameters": executor.env.scene.to_dict(),
                "replans": outcome.replans, "skill_failures": outcome.skill_failures,
                "physical_initializations": executor.env.reset_count, "skill_switch_resets": 0,
                "elapsed": float(executor.env.data.time),
                "final_robot_position": executor.env.data.xpos[executor.env.base_body_id].tolist(),
                "records": [{"skill": record.action.skill.name, "object_id": record.action.object_id, "support_id": record.action.support_id, "target_pose": record.action.target_pose.tolist(), "status": record.status.value, "execution_steps": record.steps} for record in outcome.records],
                "stages": stages,
                "transition_count": len(transitions),
                "event_sample_count": sum(transition.event_sample_count for transition in transitions),
            }
            if references is not None:
                expected = references[seed]
                result["reference_match"] = (result["succeeded"] == expected["succeeded"] and stages == expected["stages"] and result["elapsed"] == expected["elapsed"] and result["final_robot_position"] == expected["final_robot_position"])
            results.append(result)
            print(f"BOX_EXECUTOR case={case_name} seed={seed} success={outcome.succeeded} reason={outcome.reason} skills={[record.action.skill.name for record in outcome.records]} reference_match={result.get('reference_match')}", flush=True)
    finally:
        if writer is not None:
            writer.close()
    regression_passed = all(row["reference_match"] for row in results) if references is not None else None
    report = {
        "schema_version": 1, "driver": "reconfigurable_executor_box_support",
        "policy_sha256": policies, "runtime_provenance": provenance,
        "successful_episodes": sum(row["succeeded"] for row in results), "evaluated_episodes": len(results),
        "transition_count": transition_count, "full_snapshot_available": args.snapshot_dir is not None and not args.snapshot_first_only,
        "snapshot_scope": "first_skill" if args.snapshot_first_only else "every_skill",
        "record_jsonl": str(args.record_jsonl) if args.record_jsonl is not None else None,
        "reference_sha256": hashlib.sha256(args.reference_json.read_bytes()).hexdigest() if args.reference_json is not None else None,
        "regression_passed": regression_passed, "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(f"Box executor: {report['successful_episodes']}/{len(results)}, transitions={transition_count}, regression_passed={regression_passed}")
    passed = regression_passed if regression_passed is not None else all(row["succeeded"] for row in results)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()