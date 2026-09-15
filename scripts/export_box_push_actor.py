"""Export the validated hybrid PUSH actor without replacing deployed policies."""

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import shutil

import torch


ROOT = Path(__file__).resolve().parents[1]
MDP = ROOT / "source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/mdp"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation-json", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Output directory must not already exist")
    checkpoint_hash = sha256(args.checkpoint)
    evaluations = []
    for path in args.evaluation_json:
        result = json.loads(path.read_text())
        if result["checkpoint_sha256"] != checkpoint_hash or result["task"] != "GO2-ARX5-Box-Push-Hybrid-Play":
            raise ValueError("Evaluation does not belong to this hybrid PUSH checkpoint")
        if result["requested_episodes"] < 32 or result["incomplete_episodes"] or result["successful_episodes"] / result["requested_episodes"] < 29 / 32:
            raise ValueError("Evaluation has not passed the independent success gate")
        for name, expected in result["control_sha256"].items():
            if Path(name).name != name or sha256(MDP / name) != expected:
                raise ValueError(f"Controller source differs from evaluation: {name}")
        evaluations.append({"file": path.name, "sha256": sha256(path), "result": result})
    load_actor = runpy.run_path(str(ROOT / "scripts/rsl_rl/initialization.py"))["actor_from_state_dict"]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    actor = load_actor(checkpoint["actor_state_dict"])
    if (actor[0].in_features, actor[-1].out_features) != (226, 12):
        raise ValueError("Expected hybrid PUSH dimensions 226 -> 12")
    scripted = torch.jit.script(actor)
    generator = torch.Generator().manual_seed(0)
    observations = torch.cat((torch.zeros(1, 226), torch.randn(128, 226, generator=generator)), dim=0)
    with torch.inference_mode():
        reference = actor(observations)
        actual = scripted(observations)
        torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-6)
    params = args.checkpoint.parent / "params"
    parameter_hashes = {name: sha256(params / name) for name in ("env.yaml", "agent.yaml")}
    args.output_dir.mkdir(parents=True)
    policy_path = args.output_dir / "policy.pt"
    scripted.save(str(policy_path))
    restored = torch.jit.load(str(policy_path), map_location="cpu").eval()
    with torch.inference_mode():
        torch.testing.assert_close(restored(observations), reference, rtol=1e-6, atol=1e-6)
    for name in parameter_hashes:
        shutil.copyfile(params / name, args.output_dir / name)
    for evaluation in evaluations:
        with (args.output_dir / evaluation["file"]).open("x") as stream:
            json.dump(evaluation["result"], stream, indent=2, allow_nan=False)
    manifest = {
        "schema_version": 1, "task": "GO2-ARX5-Box-Push-Hybrid-Play",
        "checkpoint_sha256": checkpoint_hash, "policy_sha256": sha256(policy_path),
        "actor_input_dimension": 226, "actor_output_dimension": 12,
        "actor_observation_groups": ["policy", "box"],
        "policy_history_dimension": 210, "box_and_controller_state_dimension": 16,
        "actor_outputs": "12 leg actions; arm commands are produced by PushArmIK",
        "action_history": "18 combined joint commands from BoxPushLegAction",
        "required_controller": "BoxPushLegAction with PushArmIK, not the legacy 210-input runtime",
        "mujoco_physical_validation_passed": False,
        "control_sha256": evaluations[0]["result"]["control_sha256"],
        "training_parameters_sha256": parameter_hashes,
        "export_max_abs_error": float((actual - reference).abs().max()),
        "export_checked_observations": len(observations),
        "evaluations": [{"file": item["file"], "sha256": item["sha256"]} for item in evaluations],
    }
    with (args.output_dir / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()