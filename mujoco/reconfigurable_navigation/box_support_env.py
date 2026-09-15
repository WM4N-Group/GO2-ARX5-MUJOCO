"""Accepted box-support scene exposed through the project observation interface."""

from pathlib import Path

import mujoco
import numpy as np

from .box_push_runtime import BoxPushRuntime
from .box_robot_profile import apply_box_robot_profile
from .box_support_control import surface_entry
from .box_support_scene import BoxSupportScene
from .env import BlockedPassageEnv
from .locomotion_runtime import JOINT_NAMES, projected_gravity
from .representations import Capability, ObjectState, ObjectType, OracleObservation, SkillType


ROOT = Path(__file__).resolve().parents[1]
BOX_ID = 10
PLATFORM_ID = 20


def make_model(height=0.20, mass=5.0, friction=0.4, platform_height=None, *, scene=None):
    if scene is not None:
        height, mass, friction = scene.box_size[2], scene.box_mass, scene.box_friction
        platform_height = scene.platform_size[2]
    if not all(np.isfinite(value) and value > 0.0 for value in (height, mass, friction)):
        raise ValueError("Physical parameters must be finite and positive")
    spec = mujoco.MjSpec.from_file(str(ROOT / "robots/go2_arx5/go2_arx5.xml"))
    apply_box_robot_profile(spec)
    spec.worldbody.add_light(pos=[0.0, -3.0, 5.0], dir=[0.0, 0.0, -1.0], diffuse=[0.8, 0.8, 0.8], ambient=[0.35, 0.35, 0.35])
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0.0, 0.0, 0.05], friction=[0.6, 0.0, 0.0], priority=1, condim=3, conaffinity=3)
    box_pose = (1.10, 0.0, 0.0) if scene is None else scene.box_pose
    box_size = np.array([1.2, 1.2, height] if scene is None else scene.box_size)
    box = spec.worldbody.add_body(name="push_box", pos=[*box_pose[:2], height / 2.0], quat=[np.cos(box_pose[2] / 2.0), 0.0, 0.0, np.sin(box_pose[2] / 2.0)])
    box.add_freejoint(name="push_box_joint")
    box.add_geom(name="push_box", type=mujoco.mjtGeom.mjGEOM_BOX, size=box_size / 2.0, mass=mass, friction=[friction, 0.0, 0.0], priority=2, condim=3, conaffinity=3, rgba=[0.65, 0.3, 0.15, 1.0])
    if platform_height is not None:
        if not np.isfinite(platform_height) or platform_height <= height:
            raise ValueError("Platform must be higher than the box")
        platform_pose = (3.3, 0.0, 0.0) if scene is None else scene.platform_pose
        platform_size = np.array([2.0, 1.6, platform_height] if scene is None else scene.platform_size)
        spec.worldbody.add_geom(name="high_platform", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[*platform_pose[:2], platform_height / 2.0], size=platform_size / 2.0, friction=[0.6, 0.0, 0.0], priority=1, condim=3, conaffinity=3, rgba=[0.25, 0.55, 0.72, 1.0])
    model = spec.compile()
    for name in JOINT_NAMES:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        dof = model.jnt_dofadr[joint]
        model.dof_damping[dof] = 0.0
        model.dof_frictionloss[dof] = 0.01
        model.dof_armature[dof] = 0.01
    return model


def make_episode(policy, seed, platform_height=None, *, scene=None):
    model = make_model(platform_height=platform_height, scene=scene)
    data = mujoco.MjData(model)
    box = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "push_box")
    runtime = BoxPushRuntime(policy, model=model, data=data, box_geom=box, goal=np.array([1.70, 0.0, 0.10]))
    generator = np.random.default_rng(seed)
    data.qpos[:3] = [*generator.uniform(-0.03, 0.03, size=2), 0.33]
    yaw = generator.uniform(-0.04, 0.04)
    if scene is not None:
        data.qpos[:2] += scene.robot_pose[:2]
        yaw += scene.robot_pose[2]
    data.qpos[3:7] = [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]
    joint_ids = model.dof_jntid[runtime.joint_dof_adr]
    midpoint = model.jnt_range[joint_ids].mean(axis=1)
    half_range = 0.45 * np.ptp(model.jnt_range[joint_ids], axis=1)
    data.qpos[runtime.joint_qpos_adr] = np.clip(runtime.default_qpos * generator.uniform(0.95, 1.05, 18), midpoint - half_range, midpoint + half_range)
    box_qpos = int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "push_box_joint")])
    data.qpos[box_qpos:box_qpos + 2] += generator.uniform(-0.03, 0.03, 2)
    yaw = generator.uniform(-0.04, 0.04)
    if scene is not None:
        yaw += scene.box_pose[2]
    data.qpos[box_qpos + 3:box_qpos + 7] = [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]
    mujoco.mj_forward(model, data)
    start = data.geom_xpos[box].copy()
    runtime.arm.goal = start + [0.60, 0.0, 0.0]
    runtime.activate()
    return runtime, start


class BoxSupportEnv(BlockedPassageEnv):
    def __init__(self, runtime, seed, scene=None):
        self.model, self.data = runtime.model, runtime.data
        self.push_runtime = runtime
        self.seed = seed
        self.reset_count = 1
        self.active_skill = None
        self.capability = Capability(max_step_height=0.20, max_pushable_mass=5.0)
        self.base_body_id = runtime.base_id
        self.robot_joint_id = int(self.model.body_jntadr[self.base_body_id])
        self.robot_qpos_adr = int(self.model.jnt_qposadr[self.robot_joint_id])
        self.box_geom_id = runtime.arm.box_geom
        self.box_body_id = runtime.arm.box_body
        self.platform_geom_id = self._geom_id("high_platform")
        self.floor_geom_id = self._geom_id("floor")
        self.surface_geoms = {self.floor_geom_id, self.box_geom_id, self.platform_geom_id}
        self.object_geoms = {BOX_ID: self.box_geom_id, PLATFORM_ID: self.platform_geom_id}
        self.foot_geom_ids = frozenset(self._geom_id(name) for name in ("FL", "FR", "RL", "RR"))
        self.robot_geom_ids = frozenset(index for index, body in enumerate(self.model.geom_bodyid) if self._body_descends_from(int(body), self.base_body_id))
        self.hand_geom_ids = frozenset(index for index, body in enumerate(self.model.geom_bodyid) if body in runtime.arm.hand_bodies)
        platform = self.data.geom_xpos[self.platform_geom_id]
        edge = platform[0] - self.model.geom_size[self.platform_geom_id, 0]
        self.placement_target = np.array([edge - runtime.arm.box_size[0] / 2.0, platform[1], runtime.arm.goal[2]])
        self.push_goal = self.placement_target + [0.02, 0.0, 0.0]
        self.scene = scene or BoxSupportScene()
        self.scene_family = "box_support_nominal" if scene is None else scene.scene_family
        self.dataset_split = "integration_regression_unsplit" if scene is None else scene.dataset_split

    @property
    def push_target(self):
        return np.array([self.push_goal[0], self.push_goal[1], 0.0], dtype=np.float32)

    def reset(self, seed=None):
        raise RuntimeError("Create a new box-support episode; live skills must not reset physics")

    def observe(self):
        contacts, body_contacts = set(), set()
        illegal = False
        for contact in self.data.contact[:self.data.ncon]:
            pair = {int(contact.geom1), int(contact.geom2)}
            if pair & self.push_runtime.base_geom_ids and pair & self.surface_geoms:
                illegal = True
            for object_id, geom in self.object_geoms.items():
                if geom not in pair:
                    continue
                other = int(contact.geom2) if contact.geom1 == geom else int(contact.geom1)
                if other in self.robot_geom_ids:
                    contacts.add(object_id)
                    if other not in self.hand_geom_ids and (other not in self.foot_geom_ids or self.active_skill == SkillType.PUSH):
                        body_contacts.add(object_id)
        hand = self.push_runtime.arm.contacts()
        if self.active_skill == SkillType.PUSH:
            illegal |= hand["invalid_hand"] or hand["forbidden"]
        support = self.push_runtime.support_contacts(self.foot_geom_ids, self.surface_geoms)
        position = self.data.xpos[self.base_body_id]
        rotation = self.data.xmat[self.base_body_id].reshape(3, 3)
        linear, angular = self._body_velocity(self.base_body_id)
        valid = bool(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all() and position[2] >= 0.18)
        robot = np.zeros(12, dtype=np.float32)
        robot[:3] = position
        robot[3:6] = [np.arctan2(rotation[2, 1], rotation[2, 2]), np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0)), np.arctan2(rotation[1, 0], rotation[0, 0])]
        robot[6:9] = linear
        robot[9:] = [angular[2], len(support), float(valid)]
        objects = []
        for object_id, geom in self.object_geoms.items():
            center = self.data.geom_xpos[geom].copy()
            size = self.model.geom_size[geom].copy() * 2.0
            entry, yaw = surface_entry(self.model, self.data, geom)
            movable = object_id == BOX_ID
            linear, angular = self._body_velocity(self.box_body_id) if movable else (np.zeros(3), np.zeros(3))
            objects.append(ObjectState(
                object_id, ObjectType.MOVABLE_BOX if movable else ObjectType.PLATFORM,
                center, size, yaw=float(yaw), movable=movable, supportable=True,
                mass=float(self.model.body_mass[self.box_body_id]) if movable else 0.0,
                linear_velocity=linear, angular_velocity=angular,
                climb_entry_pose=np.array([entry[0], entry[1], yaw]),
                climb_landing_pose=np.array([center[0], center[1], center[2] + size[2] / 2.0 + 0.28, yaw]),
            ))
        platform_top = self.data.geom_xpos[self.platform_geom_id, 2] + self.model.geom_size[self.platform_geom_id, 2]
        goal = np.array([*self.data.geom_xpos[self.platform_geom_id, :2], platform_top, 1.0], dtype=np.float32)
        return OracleObservation(robot, goal, tuple(objects), self.capability, valid, tuple(sorted(contacts)), (BOX_ID,) if hand["valid"] else (), tuple(sorted(body_contacts)), bool(illegal))