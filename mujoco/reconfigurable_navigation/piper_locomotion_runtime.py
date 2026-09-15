"""GO2-PIPER locomotion initialization using the existing NAV control contract."""

from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import yaml

from .locomotion_runtime import LocomotionRuntime
from .piper_robot_profile import apply_piper_robot_profile


ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy/deploy_mujoco/go2_piper"
JOINT_NAMES = tuple(
    f"{leg}_{joint}_joint"
    for leg in ("FL", "FR", "RL", "RR")
    for joint in ("hip", "thigh", "calf")
) + tuple(f"joint{index}" for index in range(1, 7))


class PiperLocomotionRuntime(LocomotionRuntime):
    def __init__(self, config_path=DEPLOY / "config.yaml", *, seed=0, initialization="nominal", profile_path=None):
        if initialization not in ("nominal", "model"):
            raise ValueError("Unknown initialization")
        self.config_path = Path(config_path).resolve()
        with self.config_path.open() as source:
            config = yaml.safe_load(source)
        self.policy_path = Path(config["policy_path"].replace("{CURRENT_ROOT_DIR}", str(ROOT)))
        self.scene_path = Path(config["xml_path"].replace("{CURRENT_ROOT_DIR}", str(ROOT)))
        specification = mujoco.MjSpec.from_file(str(self.scene_path))
        self.profile_path = Path(profile_path).resolve() if profile_path is not None else None
        if self.profile_path is not None:
            apply_piper_robot_profile(specification, self.profile_path)
        model = specification.compile()
        data = mujoco.MjData(model)
        base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        if base_id < 0:
            raise ValueError("Missing PIPER base_link")
        root_joint = int(model.body_jntadr[base_id])
        if root_joint < 0 or model.jnt_type[root_joint] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("PIPER base must have a free joint")
        root_qpos = int(model.jnt_qposadr[root_joint])
        joint_ids = np.array([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in JOINT_NAMES
        ])
        if np.any(joint_ids < 0) or not np.array_equal(model.actuator_trnid[:, 0], joint_ids):
            raise ValueError("PIPER actuator/joint mapping does not match the configuration")
        if initialization == "nominal":
            generator = np.random.default_rng(seed)
            yaw = generator.uniform(-0.15, 0.15)
            data.qpos[root_qpos:root_qpos + 3] = [0.0, 0.0, 0.35]
            data.qpos[root_qpos + 3:root_qpos + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
            angles = np.asarray(config["default_angles"]) + generator.uniform(-0.015, 0.015, 18)
            data.qpos[model.jnt_qposadr[joint_ids]] = np.clip(
                angles, model.jnt_range[joint_ids, 0], model.jnt_range[joint_ids, 1],
            )
        mujoco.mj_forward(model, data)
        env = SimpleNamespace(
            model=model, data=data, deploy_cfg=config,
            robot_joint_id=root_joint, robot_qpos_adr=root_qpos,
        )
        super().__init__(env, self.policy_path, joint_names=JOINT_NAMES, base_body_name="base_link")