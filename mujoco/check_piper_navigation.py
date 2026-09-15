"""Evaluate PIPER target-pose navigation with the existing NavigateSkill."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from check_piper_locomotion import PhysicsMonitor, RolloutFailure
from check_piper_startup import sha256
from reconfigurable_navigation.piper_locomotion_runtime import PiperLocomotionRuntime
from reconfigurable_navigation.piper_robot_profile import PROFILE_PATH
from reconfigurable_navigation.representations import Capability, OracleObservation, SkillAction, SkillType
from reconfigurable_navigation.skills.navigate import NavigateConfig, NavigateSkill, wrap_angle
from reconfigurable_navigation.skills.base import SkillStatus


def observation(runtime, target):
    state = np.zeros(12, dtype=np.float32)
    state[:3] = runtime.data.qpos[:3]
    rotation = runtime.data.xmat[runtime.base_id].reshape(3, 3)
    state[5] = np.arctan2(rotation[1, 0], rotation[0, 0])
    return OracleObservation(state, target, (), Capability(), state_valid=np.isfinite(state).all())


def run_episode(seed):
    runtime = PiperLocomotionRuntime(seed=seed, profile_path=PROFILE_PATH)
    generator = np.random.default_rng(seed)
    heading = generator.uniform(-np.pi, np.pi)
    distance = generator.uniform(1.0, 2.0)
    target = np.array([distance * np.cos(heading), distance * np.sin(heading), generator.uniform(-np.pi, np.pi)], dtype=np.float32)
    monitor = PhysicsMonitor(runtime, np.zeros(3))
    runtime.on_physics_step = monitor
    skill = NavigateSkill(NavigateConfig(position_tolerance=0.07, yaw_tolerance=0.08))
    action = SkillAction(SkillType.NAV, target)
    reason = "completed"
    try:
        for _step in range(round(3.0 / runtime.control_dt)):
            runtime.hold_default()
        if not skill.can_execute(observation(runtime, target), action):
            raise RolloutFailure("navigation_rejected")
        skill.reset(action)
        monitor.stage = "navigation"
        for _step in range(skill.config.timeout_steps):
            command = skill.step(observation(runtime, target))
            runtime.step(command.velocity, command.end_effector_pose)
            if skill.status != SkillStatus.RUNNING:
                break
        if skill.status != SkillStatus.SUCCEEDED:
            raise RolloutFailure(skill.failure_reason or "navigation_timeout")
        monitor.stage = "settle"
        for _step in range(round(1.0 / runtime.control_dt)):
            runtime.step(np.zeros(3))
        final = observation(runtime, target)
        if np.linalg.norm(final.robot_state[:2] - target[:2]) > 0.12 or abs(wrap_angle(float(final.robot_state[5] - target[2]))) > 0.15:
            raise RolloutFailure("goal_drift_after_stop")
        if np.linalg.norm(runtime.data.qvel[:2]) > 0.10 or abs(runtime.data.qvel[5]) > 0.15:
            raise RolloutFailure("not_stopped")
    except RolloutFailure as error:
        reason = str(error)
    final = observation(runtime, target)
    return {
        "seed": seed, "passed": reason == "completed", "reason": reason,
        "target": target.tolist(), "final_pose": [*final.robot_state[:2].tolist(), float(final.robot_state[5])],
        "position_error": float(np.linalg.norm(final.robot_state[:2] - target[:2])),
        "yaw_error": abs(wrap_angle(float(final.robot_state[5] - target[2]))),
        "elapsed_seconds": float(runtime.data.time), "physical_initializations": 1,
        "min_base_height": monitor.min_height, "max_tilt_rad": monitor.max_tilt,
        "first_illegal_contact": monitor.first_illegal_contact,
        "policy_sha256": sha256(runtime.policy_path), "config_sha256": sha256(runtime.config_path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.seeds < 1 or args.seed_offset < 0 or args.output_json.exists():
        parser.error("Require positive seeds, nonnegative seed offset and a new output path")
    torch.set_num_threads(1)
    results = []
    for seed in range(args.seed_offset, args.seed_offset + args.seeds):
        result = run_episode(seed)
        results.append(result)
        print(f"seed={seed} passed={result['passed']} reason={result['reason']}", flush=True)
    root = Path(__file__).parent
    paths = [Path(__file__), PROFILE_PATH, root / "check_piper_locomotion.py"]
    paths.extend(root / "reconfigurable_navigation" / name for name in ("piper_robot_profile.py", "piper_locomotion_runtime.py", "locomotion_runtime.py", "skills/navigate.py"))
    paths.extend(path for path in (root / "robots/go2_piper").rglob("*") if path.is_file())
    report = {
        "schema_version": 1, "scope": "flat-floor target-pose NAV with privileged state; no obstacle-planner or hardware acceptance",
        "controller_tolerances": {"position": 0.07, "yaw": 0.08},
        "acceptance_tolerances": {"position": 0.12, "yaw": 0.15, "hold_seconds": 1.0},
        "successful_episodes": sum(result["passed"] for result in results), "evaluated_episodes": len(results),
        "failures": dict(Counter(result["reason"] for result in results if not result["passed"])),
        "source_assets_sha256": {str(path): sha256(path) for path in paths}, "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x") as destination:
        json.dump(report, destination, indent=2, allow_nan=False)
    print(json.dumps({key: report[key] for key in ("successful_episodes", "evaluated_episodes", "failures")}, indent=2))
    return 0 if all(result["passed"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())