"""Run the physical NAV-PUSH-NAV-CLIMB-NAV MVP course."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import time
import uuid

import mujoco
import mujoco.viewer
import numpy as np

from reconfigurable_navigation.climb_runtime import ClimbRuntime
from reconfigurable_navigation.complex_course_env import ComplexCourseEnv
from reconfigurable_navigation.data import SkillTransition
from reconfigurable_navigation.data.snapshot import SimulatorSnapshot
from reconfigurable_navigation.locomotion_runtime import LocomotionRuntime
from reconfigurable_navigation.occupancy import GridConfig
from reconfigurable_navigation.oracle_planner import OraclePlanner
from reconfigurable_navigation.representations import (
    OracleObservation,
    SkillAction,
    SkillType,
)
from reconfigurable_navigation.runtime import (
    OracleReplanner,
    ReconfigurableExecutor,
)
from reconfigurable_navigation.skills import Skill


def run_episode(
    seed: int,
    visualize: bool = False,
    realtime: bool = False,
    record_jsonl: Path | None = None,
    snapshot_dir: Path | None = None,
) -> bool:
    if snapshot_dir is not None and record_jsonl is None:
        raise ValueError("snapshot_dir requires record_jsonl")
    env = ComplexCourseEnv()
    env.reset(seed=seed)
    locomotion_runtime = LocomotionRuntime(env)
    climb_runtime = ClimbRuntime(
        model=env.model,
        data=env.data,
        deploy_config=env.deploy_cfg,
    )
    replanner = OracleReplanner(
        env.push_target,
        planner=OraclePlanner(GridConfig(size=18.0)),
    )
    executor = ReconfigurableExecutor(
        env,
        locomotion_runtime,
        climb_runtime=climb_runtime,
        replanner=replanner,
    )
    fingertip_contact = False
    body_contact = False
    episode_id = uuid.uuid4().hex if record_jsonl is not None else None
    transition_index = 0
    snapshot_metadata = {}
    if record_jsonl is not None:
        record_jsonl.parent.mkdir(parents=True, exist_ok=True)
    viewer = None
    if visualize:
        viewer = mujoco.viewer.launch_passive(env.model, env.data)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = env.base_body_id
        viewer.cam.distance = 5.0
        viewer.cam.azimuth = 145.0
        viewer.cam.elevation = -25.0

    def on_step(
        action: SkillAction | None,
        _skill: Skill | None,
        observation: OracleObservation,
    ) -> bool:
        nonlocal fingertip_contact, body_contact
        if action is not None and action.skill == SkillType.PUSH:
            fingertip_contact = fingertip_contact or (
                action.object_id in observation.end_effector_contact_object_ids
            )
            body_contact = body_contact or (
                action.object_id in observation.body_contact_object_ids
            )
        if viewer is not None:
            if not viewer.is_running():
                return False
            viewer.sync()
        if realtime:
            time.sleep(locomotion_runtime.control_dt)
        return True

    def on_skill_start(_action: SkillAction, previous_skill: SkillType | None) -> None:
        assert snapshot_dir is not None and record_jsonl is not None
        snapshot_path = snapshot_dir / f"{episode_id}-{transition_index}.snapshot"
        snapshot = SimulatorSnapshot.capture(executor, previous_skill)
        snapshot.save(snapshot_path)
        snapshot_metadata.clear()
        snapshot_metadata.update(
            snapshot_file=os.path.relpath(snapshot_path.resolve(), record_jsonl.parent.resolve()),
            snapshot_sha256=hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
        )

    def on_transition(transition: SkillTransition) -> None:
        nonlocal transition_index
        assert record_jsonl is not None
        line = transition.to_json(
            {
                "episode_id": episode_id,
                "scenario": "complex_course",
                "seed": seed,
                "skill_index": transition_index,
                **snapshot_metadata,
            }
        )
        with record_jsonl.open("a", encoding="utf-8") as output:
            output.write(line + "\n")
        transition_index += 1

    result = executor.run(
        on_step=on_step,
        on_event=print if visualize else None,
        on_transition=on_transition if record_jsonl is not None else None,
        on_skill_start=on_skill_start if snapshot_dir is not None else None,
    )
    observation = env.observe()
    skills = tuple(record.action.skill for record in result.records)
    phases = tuple(
        skill
        for index, skill in enumerate(skills)
        if index == 0 or skill != skills[index - 1]
    )
    expected = (
        SkillType.NAV,
        SkillType.PUSH,
        SkillType.NAV,
        SkillType.CLIMB,
        SkillType.NAV,
    )
    succeeded = bool(
        result.succeeded
        and phases == expected
        and env.reset_count == 1
        and fingertip_contact
        and not body_contact
        and not observation.illegal_collision
    )
    print(
        f"seed={seed} success={succeeded} reason={result.reason} "
        f"phases={','.join(skill.name for skill in phases)} "
        f"records={len(result.records)} replans={result.replans} "
        f"resets={env.reset_count} fingertip_contact={fingertip_contact} "
        f"body_contact={body_contact} illegal_collision={observation.illegal_collision} "
        f"robot=({observation.robot_state[0]:.3f}, "
        f"{observation.robot_state[1]:.3f}, {observation.robot_state[2]:.3f})"
    )
    if viewer is not None and viewer.is_running():
        print("Close the Viewer window to exit.")
        while viewer.is_running():
            viewer.sync()
            time.sleep(1.0 / 60.0)
    if viewer is not None:
        viewer.close()
    return succeeded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument(
        "--record-jsonl",
        type=Path,
        help="Append skill-boundary observations to a JSONL file.",
    )
    parser.add_argument(
        "--snapshot-dir", type=Path,
        help="Save trusted, version-specific skill-start snapshots referenced by JSONL.",
    )
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    if args.visualize and args.seeds != 1:
        parser.error("--visualize requires --seeds 1")
    if args.snapshot_dir is not None and args.record_jsonl is None:
        parser.error("--snapshot-dir requires --record-jsonl")
    successes = sum(
        run_episode(
            seed,
            visualize=args.visualize,
            realtime=args.visualize and not args.no_realtime,
            record_jsonl=args.record_jsonl,
            snapshot_dir=args.snapshot_dir,
        )
        for seed in range(args.seeds)
    )
    print(f"Complex course evaluation: {successes}/{args.seeds} succeeded")
    raise SystemExit(0 if successes == args.seeds else 1)


if __name__ == "__main__":
    main()