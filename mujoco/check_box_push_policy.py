"""Strict physical check of the experimental MuJoCo hybrid PUSH runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

import mujoco

from reconfigurable_navigation.box_robot_profile import box_runtime_provenance
from reconfigurable_navigation.box_support_control import push_box
from reconfigurable_navigation.box_support_env import make_episode, make_model


ROOT = Path(__file__).resolve().parent


def run_episode(policy, seed, duration=12.0, *, runtime=None):
    if runtime is None:
        runtime, _start = make_episode(policy, seed)
    return push_box(runtime, seed, duration)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.seeds < 1 or args.seed_offset < 0 or args.output_json.exists():
        parser.error("Use positive seed count, nonnegative offset and a new output file")
    torch.set_num_threads(1)
    results = [run_episode(args.policy, seed) for seed in range(args.seed_offset, args.seed_offset + args.seeds)]
    successes = sum(result["succeeded"] for result in results)
    passed = successes / len(results) >= 29 / 32
    paths = [Path(__file__), ROOT / "reconfigurable_navigation/box_push_runtime.py", ROOT / "reconfigurable_navigation/climb_runtime.py"]
    report = {"schema_version": 1, "policy_sha256": hashlib.sha256(args.policy.read_bytes()).hexdigest(), "code_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}, "mujoco_version": mujoco.__version__, "fixed_support": False, "box_size": [1.2, 1.2, 0.20], "box_mass": 5.0, "box_friction": 0.4, "push_distance": 0.60, "passed": passed, "results": results}
    report["runtime_provenance"] = box_runtime_provenance()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(f"MuJoCo hybrid PUSH: {successes}/{len(results)}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()