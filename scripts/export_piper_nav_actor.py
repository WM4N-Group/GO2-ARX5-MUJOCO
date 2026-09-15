"""Package the unchanged PIPER NAV actor with verified deployment evidence."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal-report", type=Path, required=True)
    parser.add_argument("--native-reports", type=Path, nargs="+", required=True)
    parser.add_argument("--alignment-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Output bundle must not exist")
    policy = ROOT / "mujoco/deploy/policy/go2_piper/policy.pt"
    config = ROOT / "mujoco/deploy/deploy_mujoco/go2_piper/config.yaml"
    profile = ROOT / "mujoco/deploy/piper_robot_profile.json"
    goal = json.loads(args.goal_report.read_text())
    if goal["evaluated_episodes"] < 100 or goal["successful_episodes"] / goal["evaluated_episodes"] < 0.95:
        raise ValueError("NAV goal evaluation gate not met")
    for row in goal["results"]:
        if row["policy_sha256"] != digest(policy) or row["config_sha256"] != digest(config):
            raise ValueError("NAV policy or configuration differs from evaluation")
    for name, expected in goal["source_assets_sha256"].items():
        if digest(name) != expected:
            raise ValueError(f"NAV deployment source changed: {name}")
    for path in args.native_reports:
        native = json.loads(path.read_text())
        if native["task"] != "GO2-PIPER-Flat" or native["policy_sha256"] != digest(policy) or native["successful_episodes"] != native["num_envs"]:
            raise ValueError("Native NAV reference does not validate this policy")
    alignment = json.loads(args.alignment_report.read_text())
    if not alignment["passed"] or alignment["profile_sha256"] != digest(profile):
        raise ValueError("Robot profile alignment is missing or changed")
    paths = [args.goal_report, args.alignment_report, *args.native_reports]
    if len({path.name for path in paths}) != len(paths):
        raise ValueError("Report filenames must be distinct")
    args.output_dir.mkdir(parents=True)
    for source, filename in ((policy, "policy.pt"), (config, "deploy.yaml"), (profile, "robot_profile.json")):
        shutil.copyfile(source, args.output_dir / filename)
    for path in paths:
        shutil.copyfile(path, args.output_dir / path.name)
    manifest = {
        "schema_version": 1, "robot": "go2_piper", "skill": "NAV", "retrained": False,
        "policy_sha256": digest(policy), "deploy_config_sha256": digest(config), "robot_profile_sha256": digest(profile),
        "actor_input_dimension": 210, "actor_output_dimension": 18,
        "scope": "privileged-state flat-floor goal navigation with legacy direct-PD control and aligned inertial profile",
        "native_reference_passed": True, "mujoco_goal_validation_passed": True,
        "hardware_validation_passed": False, "obstacle_planner_validation_passed": False,
        "controller_tolerances": goal["controller_tolerances"], "acceptance_tolerances": goal["acceptance_tolerances"],
        "goal_successes": goal["successful_episodes"], "goal_episodes": goal["evaluated_episodes"],
        "reports": [{"file": path.name, "sha256": digest(path)} for path in paths],
        "source_assets_sha256": goal["source_assets_sha256"],
    }
    with (args.output_dir / "manifest.json").open("x") as destination:
        json.dump(manifest, destination, indent=2, allow_nan=False)
    print(json.dumps({"bundle": str(args.output_dir), "policy_sha256": digest(policy), "goal_successes": goal["successful_episodes"], "goal_episodes": goal["evaluated_episodes"]}, indent=2))


if __name__ == "__main__":
    main()