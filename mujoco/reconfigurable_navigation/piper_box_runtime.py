"""PIPER box-skill scene and runtimes with explicit robot configuration."""

from pathlib import Path

import mujoco
import numpy as np
import yaml

from .box_climb_runtime import BoxClimbRuntime
from .box_push_runtime import BoxPushRuntime
from .piper_locomotion_runtime import DEPLOY, JOINT_NAMES, ROOT
from .piper_robot_profile import apply_piper_collision_profile, apply_piper_robot_profile


def make_episode(skill, policy, seed, height=0.20, *, push_ik_mode="position", native_collisions=False):
    if skill not in ("push", "climb") or not np.isfinite(height) or height <= 0:
        raise ValueError("Expected PUSH/CLIMB and a positive height")
    if push_ik_mode not in ("position", "pose"):
        raise ValueError("Unknown PUSH IK mode")
    spec = mujoco.MjSpec.from_file(str(ROOT / "robots/go2_piper/go2piper.xml"))
    for mesh in spec.meshes:
        mesh.file = str(ROOT / "robots/go2_piper/assets" / Path(mesh.file).name)
    apply_piper_robot_profile(spec)
    if native_collisions:
        apply_piper_collision_profile(spec)
    for geom in spec.geoms:
        geom.group = 3 if geom.contype or geom.conaffinity else 2
    robot_bodies = [body.name for body in spec.bodies if body.name and body.name != "world"]
    for index, first in enumerate(robot_bodies):
        for second in robot_bodies[index + 1:]:
            spec.add_exclude(bodyname1=first, bodyname2=second)
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], friction=[0.6, 0, 0], priority=1, condim=3)
    if skill == "push":
        box = spec.worldbody.add_body(name="push_box", pos=[1.10, 0.0, height / 2])
        box.add_freejoint(name="push_box_joint")
        box.add_geom(name="push_box", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.6, 0.6, height / 2], mass=5.0, friction=[0.4, 0, 0], priority=2, condim=3)
    else:
        spec.worldbody.add_geom(name="support_box", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[1.35, 0, height / 2], size=[0.6, 0.6, height / 2], friction=[0.6, 0, 0], priority=1, condim=3)
    model = spec.compile()
    data = mujoco.MjData(model)
    with (DEPLOY / "config.yaml").open() as source:
        config = yaml.safe_load(source)
    config["kps"][:12] = [30.0] * 12
    config["kds"][:12] = [0.6] * 12
    options = {"model": model, "data": data, "deploy_config": config, "joint_names": JOINT_NAMES, "base_body_name": "base_link"}
    if skill == "push":
        box_id = model.geom("push_box").id
        runtime = BoxPushRuntime(policy, box_geom=box_id, goal=np.array([1.7, 0, height / 2]), arm_config={
            "hand_body_name": "end_effector", "mount_body_name": "Piper",
            "hand_body_names": ("link6", "link7", "link8"), "hand_offset": (0, 0, 0),
            "align_hand_orientation": push_ik_mode == "pose", "hand_pitch": 0.8 if push_ik_mode == "pose" else 0.0,
            "approach_distance": 0.03 if push_ik_mode == "pose" else 0.10,
        }, **options)
    else:
        runtime = BoxClimbRuntime(policy, linear_velocity_at_com=True, **options)
    runtime.joint_velocity_limits[12:] = 3.0
    generator = np.random.default_rng(seed)
    span = (0.03, 0.03, 0.04) if skill == "push" else (0.10, 0.08, 0.06)
    data.qpos[:3] = [generator.uniform(-span[0], span[0]), generator.uniform(-span[1], span[1]), 0.33]
    yaw = generator.uniform(-span[2], span[2])
    data.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    joint_ids = model.dof_jntid[runtime.joint_dof_adr]
    midpoints = model.jnt_range[joint_ids].mean(axis=1)
    half_ranges = np.ptp(model.jnt_range[joint_ids], axis=1) * 0.45
    spread = 0.05 if skill == "push" else 0.1
    data.qpos[runtime.joint_qpos_adr] = np.clip(runtime.default_qpos * generator.uniform(1-spread, 1+spread, 18), midpoints-half_ranges, midpoints+half_ranges)
    if skill == "push":
        box_position = data.joint("push_box_joint").qpos
        box_position[:2] += generator.uniform(-0.03, 0.03, 2)
        yaw = generator.uniform(-0.04, 0.04)
        box_position[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    mujoco.mj_forward(model, data)
    if skill == "push":
        runtime.arm.goal = data.geom_xpos[box_id].copy() + [0.6, 0, 0]
    runtime.activate()
    return runtime