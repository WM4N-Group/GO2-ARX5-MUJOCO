"""Export a provenance-bound box CLIMB actor or an explicit diagnostic candidate."""

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import shutil

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
MDP = ROOT / "source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/mdp"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepared_start_sources(checkpoint, evaluations):
    config = yaml.load((checkpoint.parent / "params/env.yaml").read_text(), Loader=yaml.BaseLoader)
    sources = []
    training_path = config.get("prepared_start_path", "")
    if training_path:
        sources.append({"role": "training", "path": Path(training_path), "phase": config["prepared_start_phase"]})
    for report in evaluations:
        physics = report["actual_physics"]
        if physics.get("prepared_start_path"):
            path = Path(physics["prepared_start_path"])
            if digest(path) != physics["prepared_start_sha256"]:
                raise ValueError("Prepared evaluation states changed since validation")
            sources.append({"role": "evaluation", "path": path, "phase": physics["prepared_start_phase"]})
    for source in sources:
        source["sha256"] = digest(source["path"])
        source["file"] = f"prepared-starts-{source['sha256'][:16]}.json"
    return sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation-json", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Output directory must not already exist")
    checkpoint_hash = digest(args.checkpoint)
    evaluations = []
    for path in args.evaluation_json:
        report = json.loads(path.read_text())
        if report["task"] != "GO2-ARX5-Box-Climb-Play" or report["checkpoint_sha256"] != checkpoint_hash:
            raise ValueError("Evaluation does not match this CLIMB checkpoint")
        for name, expected in report["control_sha256"].items():
            if Path(name).name != name or digest(MDP / name) != expected:
                raise ValueError(f"Evaluation controller differs: {name}")
        evaluations.append(report)
    passed = all(report["requested_episodes"] >= 32 and not report["incomplete_episodes"] and report["successful_episodes"] / report["requested_episodes"] >= 29 / 32 for report in evaluations)
    if not passed and not args.diagnostic:
        raise ValueError("Independent success gate failed; only diagnostic export is allowed")
    start_sources = prepared_start_sources(args.checkpoint, evaluations)
    load_actor = runpy.run_path(str(ROOT / "scripts/rsl_rl/initialization.py"))["actor_from_state_dict"]
    actor = load_actor(torch.load(args.checkpoint, map_location="cpu", weights_only=False)["actor_state_dict"])
    if (actor[0].in_features, actor[-1].out_features) != (253, 18):
        raise ValueError("Expected box CLIMB dimensions 253 -> 18")
    inputs = torch.cat((torch.zeros(1, 253), torch.randn(128, 253, generator=torch.Generator().manual_seed(0))))
    scripted = torch.jit.script(actor)
    with torch.inference_mode():
        expected = actor(inputs)
        actual = scripted(inputs)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    args.output_dir.mkdir(parents=True)
    scripted.save(str(args.output_dir / "policy.pt"))
    with torch.inference_mode():
        torch.testing.assert_close(torch.jit.load(str(args.output_dir / "policy.pt"))(inputs), expected, rtol=1e-6, atol=1e-6)
    params = args.checkpoint.parent / "params"
    for name in ("env.yaml", "agent.yaml"):
        shutil.copyfile(params / name, args.output_dir / name)
    for path in args.evaluation_json:
        shutil.copyfile(path, args.output_dir / path.name)
    for source in start_sources:
        destination = args.output_dir / source["file"]
        if not destination.exists():
            shutil.copyfile(source["path"], destination)
        if digest(destination) != source["sha256"]:
            raise ValueError("Prepared state copy differs from source")
    manifest = {
        "schema_version": 1, "task": "GO2-ARX5-Box-Climb-Play", "diagnostic": args.diagnostic,
        "native_validation_passed": passed, "mujoco_physical_validation_passed": False,
        "checkpoint_sha256": checkpoint_hash, "policy_sha256": digest(args.output_dir / "policy.pt"),
        "actor_input_dimension": 253, "actor_output_dimension": 18,
        "control_sha256": evaluations[0]["control_sha256"],
        "training_parameters_sha256": {name: digest(params / name) for name in ("env.yaml", "agent.yaml")},
        "evaluations": [{"file": path.name, "sha256": digest(path)} for path in args.evaluation_json],
        "prepared_start_sources": [{key: value for key, value in source.items() if key != "path"} for source in start_sources],
        "export_checked_observations": len(inputs), "export_max_abs_error": float((actual - expected).abs().max()),
    }
    with (args.output_dir / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()