"""Collect object-model transitions at live skill boundaries using isolated forks."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
import multiprocessing
from pathlib import Path
import time

import mujoco
import numpy as np
import torch

from reconfigurable_navigation.box_robot_profile import box_runtime_provenance
from reconfigurable_navigation.box_support_scene import BoxSupportScene, box_support_suite
from reconfigurable_navigation.data.candidates import build_candidates, candidate_partition_metadata, candidate_payload
from reconfigurable_navigation.data.snapshot import SimulatorSnapshot
from reconfigurable_navigation.data.snapshot_io import _code_fingerprint
from reconfigurable_navigation.data.suffix import candidate_continuation
from reconfigurable_navigation.runtime.box_support_backend import make_box_support_executor


def integration_state(env):
    kind = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(env.model, kind))
    mujoco.mj_getState(env.model, env.data, state, kind)
    return state


def collect_episode(job):
    torch.set_num_threads(1)
    with redirect_stdout(StringIO()):
        scene = BoxSupportScene(**job["scene"])
        executor = make_box_support_executor(*job["policies"], seed=job["seed"], scene=scene)
        episode_id = f"box-support-{scene.scene_id}-{job['seed']}"
        sources, candidates = [], []
        pending = {}

        def on_start(action, previous):
            index = len(sources)
            observation = executor.env.observe()
            snapshot = SimulatorSnapshot.capture(executor, previous)
            original_state = integration_state(executor.env)
            active = executor.skill_backend.active_physics
            identity = hashlib.sha256(episode_id.encode() + str(index).encode() + job["code_sha256"].encode() + original_state.tobytes()).hexdigest()
            metadata = {
                "episode_id": episode_id, "seed": job["seed"], "scene_seed": job["seed"], "skill_index": index,
                "scene_family": scene.scene_family, "scene_id": scene.scene_id, "scene_parameters": scene.to_dict(),
                "scene_case": job["case"], "dataset_split": scene.dataset_split, "split_group_id": scene.scene_family,
                "snapshot_id": identity, "snapshot_kind": "in_memory_boundary_v1", "full_snapshot_available": False,
                "physical_initializations": executor.env.reset_count, "policy_sha256": job["policy_sha256"],
                "runtime_provenance": job["runtime_provenance"],
                "start_context": {
                    "completed_operations": sorted((kind.name, object_id) for kind, object_id in executor.skill_backend.completed),
                    "proprio_schema": "box_runtime_joint_state_v1",
                    "joint_position": executor.env.data.qpos[active.joint_qpos_adr].tolist(),
                    "joint_velocity": executor.env.data.qvel[active.joint_dof_adr].tolist(),
                    "last_action": active.last_action.tolist(),
                    "actuator_target": active.target_history[-1].tolist(),
                },
            }
            if job["archive"]:
                path = Path(job["output_dir"]) / "snapshots" / f"{episode_id}-{index}.snapshot"
                snapshot.save(path)
                metadata.update(full_snapshot_available=True, snapshot_file=str(path.relative_to(job["output_dir"])), snapshot_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            seed = int(np.random.SeedSequence([job["candidate_seed"], job["seed"], index, int(scene.scene_id[:8], 16)]).generate_state(1)[0])
            proposals = build_candidates(action, count=job["count"], seed=seed, observation=observation, scene_family=scene.scene_family)
            current = []
            for candidate_index, proposal in enumerate(proposals):
                started = time.perf_counter()
                replay = snapshot.rollout(proposal.action, max_control_steps=job["max_control_steps"])
                payload = candidate_payload(proposal, replay)
                payload["continuation"] = candidate_continuation(replay, max_control_steps=job["suffix_control_steps"], max_skills=job["suffix_max_skills"])
                descriptor = json.dumps({"snapshot": identity, "action": payload["action"], "prefix_budget": job["max_control_steps"], "suffix_budget": [job["suffix_control_steps"], job["suffix_max_skills"]]}, sort_keys=True)
                payload.update(
                    candidate_id=hashlib.sha256(descriptor.encode()).hexdigest(), source_record_index=index,
                    episode_id=episode_id, scene_seed=job["seed"], scene_family=scene.scene_family, scene_id=scene.scene_id,
                    scene_parameters=scene.to_dict(), scene_case=job["case"], dataset_split=scene.dataset_split, split_group_id=scene.scene_family,
                    snapshot_id=identity, snapshot_kind="in_memory_boundary_v1", snapshot_code_sha256=job["code_sha256"],
                    candidate_index=candidate_index, candidate_seed=seed, max_control_steps=job["max_control_steps"],
                    wall_seconds=time.perf_counter() - started,
                )
                for name in ("snapshot_file", "snapshot_sha256"):
                    if name in metadata:
                        payload[name] = metadata[name]
                current.append(payload)
            np.testing.assert_array_equal(integration_state(executor.env), original_state)
            pending.update(metadata=metadata, candidates=current)

        def on_transition(transition):
            source = json.loads(transition.to_json(pending["metadata"]))
            reference = pending["candidates"][0]["transition"]
            if reference is not None and not reference["interrupted"]:
                assert {key: value for key, value in source.items() if key != "metadata"} == {key: value for key, value in reference.items() if key != "metadata"}
            sources.append(source)
            candidates.extend(pending["candidates"])

        result = executor.run(on_skill_start=on_start, on_transition=on_transition)
        for candidate in candidates:
            continuation = candidate["continuation"]
            if candidate["candidate_source"] == "reference" and continuation["oracle_task_success"] is not None:
                assert continuation["oracle_task_success"] == result.succeeded
                started_at = sources[candidate["source_record_index"]]["started_at"]
                assert np.isclose(started_at + continuation["total_sim_time"], executor.env.data.time, rtol=0.0, atol=1e-8)
        report = {"episode_id": episode_id, "scene_id": scene.scene_id, "case": job["case"], "seed": job["seed"], "split": scene.dataset_split, "succeeded": result.succeeded, "reason": result.reason, "elapsed": float(executor.env.data.time), "physical_initializations": executor.env.reset_count}
        return sources, candidates, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--push-policy", type=Path, required=True)
    parser.add_argument("--climb-policy", type=Path, required=True)
    parser.add_argument("--platform-policy", type=Path, required=True)
    parser.add_argument("--suite-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-offset", type=int, default=700)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--candidates-per-snapshot", type=int, default=6)
    parser.add_argument("--candidate-seed", type=int, default=91)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-control-steps", type=int, default=5000)
    parser.add_argument("--suffix-control-steps", type=int, default=3000)
    parser.add_argument("--suffix-max-skills", type=int, default=8)
    parser.add_argument("--archive-first-per-family", action="store_true")
    args = parser.parse_args()
    if min(args.seeds, args.workers, args.max_control_steps) < 1 or args.candidates_per_snapshot < 3 or min(args.seed_offset, args.candidate_seed, args.suffix_control_steps, args.suffix_max_skills) < 0:
        parser.error("Invalid sampling settings")
    scenes = box_support_suite() if args.suite_json is None else {name: BoxSupportScene(**values) for name, values in json.loads(args.suite_json.read_text()).items()}
    if not scenes or len({scene.scene_id for scene in scenes.values()}) != len(scenes):
        parser.error("Scene configurations must be nonempty and unique")
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    policies = [str(path.resolve()) for path in (args.push_policy, args.climb_policy, args.platform_policy)]
    hashes = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in zip(("push", "climb", "platform"), policies)}
    provenance, code = box_runtime_provenance(), _code_fingerprint()
    jobs, archived = [], set()
    for case_index, (case, scene) in enumerate(scenes.items()):
        first_seed = args.seed_offset + case_index * args.seeds
        for seed in range(first_seed, first_seed + args.seeds):
            archive = args.archive_first_per_family and scene.scene_family not in archived
            if archive:
                archived.add(scene.scene_family)
            jobs.append({"case": case, "scene": scene.to_dict(), "seed": seed, "policies": policies, "policy_sha256": hashes, "runtime_provenance": provenance, "code_sha256": code, "output_dir": str(args.output_dir.resolve()), "archive": archive, "count": args.candidates_per_snapshot, "candidate_seed": args.candidate_seed, "max_control_steps": args.max_control_steps, "suffix_control_steps": args.suffix_control_steps, "suffix_max_skills": args.suffix_max_skills})
    manifest = {"schema_version": 2, "complete": False, "source_jsonl": "transitions.jsonl", "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "candidate_generator_sha256": hashlib.sha256((Path(__file__).parent / "reconfigurable_navigation/data/candidates.py").read_bytes()).hexdigest(), "suffix_evaluator_sha256": hashlib.sha256((Path(__file__).parent / "reconfigurable_navigation/data/suffix.py").read_bytes()).hexdigest(), "code_sha256": code, "policy_sha256": hashes, "seed_offset": args.seed_offset, "seeds": args.seeds, "candidate_seed": args.candidate_seed, "candidates_per_snapshot": args.candidates_per_snapshot, "workers": args.workers, "scene_parameters": {name: scene.to_dict() for name, scene in scenes.items()}, "continuation_budget": {"max_control_steps": args.suffix_control_steps, "max_skills": args.suffix_max_skills}, "limitations": ["in-memory boundaries are not full archive hashes", "only marked representative boundaries have persisted physics archives", "single-box positive-X push scope with privileged yaw-box geometry"]}
    manifest_path = args.output_dir / "manifest.json"
    manifest["seed_assignment"] = "disjoint_contiguous_blocks_per_layout"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False))
    all_sources, reports, outcomes, continuations = [], [], Counter(), Counter()
    started = time.perf_counter()
    with (args.output_dir / "transitions.jsonl").open("x") as source_file, (args.output_dir / "candidates.jsonl").open("x") as candidate_file:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            for sources, candidates, report in pool.map(collect_episode, jobs):
                offset = len(all_sources)
                for source in sources:
                    source_file.write(json.dumps(source, allow_nan=False) + "\n")
                for candidate in candidates:
                    candidate["source_record_index"] += offset
                    candidate_file.write(json.dumps(candidate, allow_nan=False) + "\n")
                    outcomes[candidate["outcome"]] += 1
                    success = candidate["continuation"]["oracle_task_success"]
                    continuations["rejected" if not candidate["executed"] else "unknown" if success is None else "succeeded" if success else "failed"] += 1
                all_sources.extend(sources)
                reports.append(report)
                source_file.flush()
                candidate_file.flush()
                print(f"EPISODE {len(reports)}/{len(jobs)} case={report['case']} seed={report['seed']} task={report['succeeded']} starts={len(sources)} candidates={dict(outcomes)}", flush=True)
    manifest.update(candidate_partition_metadata(all_sources))
    manifest.update(complete=True, source_sha256=hashlib.sha256((args.output_dir / "transitions.jsonl").read_bytes()).hexdigest(), candidates_sha256=hashlib.sha256((args.output_dir / "candidates.jsonl").read_bytes()).hexdigest(), records=sum(outcomes.values()), executed=sum(value for name, value in outcomes.items() if name != "rejected"), source_records=len(all_sources), episodes=reports, outcomes=dict(outcomes), continuation_outcomes=dict(continuations), elapsed_wall_seconds=time.perf_counter() - started)
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False))
    print(f"OBJECT DATA COMPLETE: {manifest['records']} requests, {manifest['executed']} executions, {manifest['source_records']} starts", flush=True)


if __name__ == "__main__":
    main()