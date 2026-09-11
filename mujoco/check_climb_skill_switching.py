"""Check NAV-to-CLIMB-to-NAV switching without resetting MuJoCo state."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
import yaml

from reconfigurable_navigation.climb_runtime import ClimbRuntime, POLICY_PATH
from reconfigurable_navigation.locomotion_runtime import LocomotionRuntime
from reconfigurable_navigation.occupancy import GridConfig
from reconfigurable_navigation.oracle_planner import OraclePlanner
from reconfigurable_navigation.representations import (
    Capability,
    ObjectState,
    ObjectType,
    OracleObservation,
    SkillType,
)
from reconfigurable_navigation.runtime import (
    OracleReplanner,
    ReconfigurableExecutor,
)


ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "deploy/deploy_mujoco/go2_arx5/config.yaml"


class ClimbCourseEnv:
    """Privileged observation adapter over a shared CLIMB MuJoCo scene."""

    def __init__(self, climb_runtime: ClimbRuntime) -> None:
        self.climb_runtime = climb_runtime
        self.model = climb_runtime.model
        self.data = climb_runtime.data
        with CONFIG_PATH.open(encoding="utf-8") as config_file:
            self.deploy_cfg = yaml.safe_load(config_file)
        self.robot_joint_id = 0
        self.robot_qpos_adr = 0
        self.base_body_id = climb_runtime.base_id
        self.reset_count = 0

    @property
    def push_target(self) -> np.ndarray:
        return np.zeros(3, dtype=np.float32)

    def reset(self, seed: int | None = None) -> OracleObservation:
        self.reset_count += 1
        self.climb_runtime.reset()
        rng = np.random.default_rng(seed)
        yaw = float(rng.uniform(-0.03, 0.03))
        self.data.qpos[1] = float(rng.uniform(-0.04, 0.04))
        self.data.qpos[3:7] = [
            np.cos(yaw / 2.0),
            0.0,
            0.0,
            np.sin(yaw / 2.0),
        ]
        mujoco.mj_forward(self.model, self.data)
        return self.observe()

    def observe(self) -> OracleObservation:
        root_qpos = self.data.qpos[:7]
        yaw = np.arctan2(
            2.0
            * (root_qpos[3] * root_qpos[6] + root_qpos[4] * root_qpos[5]),
            1.0 - 2.0 * (root_qpos[5] ** 2 + root_qpos[6] ** 2),
        )
        finite = bool(
            np.isfinite(root_qpos).all()
            and np.isfinite(self.data.qvel).all()
        )
        base_contact = self.climb_runtime.base_contacts_terrain()
        state_valid = bool(finite and root_qpos[2] >= 0.18 and not base_contact)
        robot_state = np.zeros(12, dtype=np.float32)
        robot_state[:3] = root_qpos[:3]
        robot_state[5] = yaw
        robot_state[6:9] = self.data.qvel[:3]
        robot_state[9] = self.data.qvel[5]
        robot_state[11] = float(state_valid)
        platform = ObjectState(
            object_id=20,
            object_type=ObjectType.PLATFORM,
            center=np.array([4.10, 0.0, 0.08], dtype=np.float32),
            size=np.array([4.60, 4.0, 0.16], dtype=np.float32),
            supportable=True,
            climb_entry_pose=np.array([1.15, 0.0, 0.0], dtype=np.float32),
            climb_landing_pose=np.array([4.70, 0.0, 0.0], dtype=np.float32),
        )
        return OracleObservation(
            robot_state=robot_state,
            goal=np.array([6.0, 0.0, 0.0, 1.0], dtype=np.float32),
            objects=(platform,),
            capability=Capability(),
            state_valid=state_valid,
            illegal_collision=base_contact,
        )

def check_seed(seed: int, policy_path: Path) -> None:
    climb_runtime = ClimbRuntime(policy_path)
    env = ClimbCourseEnv(climb_runtime)
    env.reset(seed=seed)
    locomotion_runtime = LocomotionRuntime(env)
    replanner = OracleReplanner(
        env.push_target,
        planner=OraclePlanner(GridConfig(size=14.0)),
    )
    executor = ReconfigurableExecutor(
        env,
        locomotion_runtime,
        climb_runtime=climb_runtime,
        replanner=replanner,
    )
    result = executor.run()
    observation = env.observe()
    skills = tuple(record.action.skill for record in result.records)
    skill_phases = tuple(
        skill
        for index, skill in enumerate(skills)
        if index == 0 or skill != skills[index - 1]
    )
    expected_skills = (SkillType.NAV, SkillType.CLIMB, SkillType.NAV)
    if not result.succeeded:
        raise AssertionError(f"seed {seed} failed: {result.reason}")
    if any(record.status.value != "succeeded" for record in result.records):
        raise AssertionError(f"seed {seed} had a failed skill record")
    if skill_phases != expected_skills:
        raise AssertionError(
            f"seed {seed} executed phases {skill_phases}, expected {expected_skills}"
        )
    if env.reset_count != 1:
        raise AssertionError(f"seed {seed} reset physics {env.reset_count} times")
    if climb_runtime.base_contacts_terrain():
        raise AssertionError(f"seed {seed} ended with base-terrain contact")
    print(
        f"seed={seed} success=True phases=NAV,CLIMB,NAV "
        f"records={len(result.records)} "
        f"replans={result.replans} resets={env.reset_count} "
        f"robot=({observation.robot_state[0]:.3f}, "
        f"{observation.robot_state[1]:.3f}, {observation.robot_state[2]:.3f})"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--policy", type=Path, default=POLICY_PATH)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    for seed in range(args.seeds):
        check_seed(seed, args.policy)
    print(f"CLIMB switching evaluation: {args.seeds}/{args.seeds} succeeded")


if __name__ == "__main__":
    main()