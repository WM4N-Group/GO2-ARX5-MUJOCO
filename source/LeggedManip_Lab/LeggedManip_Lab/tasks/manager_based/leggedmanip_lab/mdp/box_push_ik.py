"""Optional IK arm controller for physical PUSH diagnostics."""

import torch

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, quat_from_euler_xyz, quat_mul

from .box_push import hand_contact, push_hand_target, push_remaining_distance
from .box_push_control import coordinated_joint_target


class PushArmIK:
    def __init__(self, env, hold_legs_until_ready=False, lock_arm_on_contact=False, align_hand_orientation=False, hand_pitch=0.0):
        self.env = env
        self.robot = env.scene["robot"]
        self.hand_id = self.robot.find_bodies("end_effector")[0][0]
        self.mount_id = self.robot.find_bodies("link0")[0][0]
        self.hand_cfg = SceneEntityCfg("robot", body_names="end_effector")
        self.hand_cfg.resolve(env.scene)
        self.lock_arm_on_contact = lock_arm_on_contact
        self.align_hand_orientation = align_hand_orientation
        self.hand_rotation = quat_from_euler_xyz(
            torch.zeros(env.num_envs, device=env.device),
            torch.full((env.num_envs,), hand_pitch, device=env.device),
            torch.zeros(env.num_envs, device=env.device),
        )
        self.locked = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env.box_push_pose_override_active = self.locked
        env.box_push_pose_override = torch.zeros((env.num_envs, 7), device=env.device)
        self.joint_ids = self.robot.find_joints("joint.*", preserve_order=True)[0]
        action_joints = self.robot.find_joints(env.cfg.actions.joint_pos.joint_names, preserve_order=True)[0]
        self.action_slots = [action_joints.index(joint_id) for joint_id in self.joint_ids]
        self.leg_slots = [index for index in range(len(action_joints)) if index not in self.action_slots]
        self.hold_legs_until_ready = hold_legs_until_ready
        self.motion_ready = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        offset = 0 if self.robot.is_fixed_base else 6
        self.jacobian_columns = [joint_id + offset for joint_id in self.joint_ids]
        self.jacobian_body = self.hand_id - 1 if self.robot.is_fixed_base else self.hand_id
        self.scale = env.cfg.actions.joint_pos.scale
        if not isinstance(self.scale, (int, float)) or self.scale <= 0.0:
            raise ValueError("IK diagnostic requires a positive scalar joint action scale")
        self.reference = self.robot.data.joint_pos[:, self.joint_ids].clone()
        self.locked_reference = self.reference.clone()
        self.stiffness = torch.stack([
            self.robot.actuators[self.robot.joint_names[joint_id]].stiffness[:, 0]
            for joint_id in self.joint_ids
        ], dim=1)
        if (self.stiffness <= 0.0).any():
            raise ValueError("Gravity compensation requires positive arm stiffness")
        self.controller = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose" if align_hand_orientation else "position", use_relative_mode=False, ik_method="dls", ik_params={"lambda_val": 0.1}),
            num_envs=env.num_envs, device=env.device,
        )

    def reset(self, env_ids):
        selected = slice(None) if env_ids is None else env_ids
        self.reference[selected] = self.robot.data.joint_pos[selected][:, self.joint_ids]
        self.motion_ready[selected] = False
        self.locked[selected] = False

    def apply(self, actions):
        position = self.robot.data.body_pos_w[:, self.hand_id]
        quaternion = self.robot.data.body_quat_w[:, self.hand_id]
        target = push_hand_target(self.env)
        if self.align_hand_orientation:
            orientation = quat_mul(self.env.scene["box"].data.root_quat_w, self.hand_rotation)
            target = torch.cat((target, orientation), dim=1)
        self.controller.set_command(target, ee_quat=quaternion)
        if self.lock_arm_on_contact:
            at_goal = push_remaining_distance(self.env) < 0.10
            newly_locked = hand_contact(self.env, self.hand_cfg) & ~self.locked & ~at_goal
            self.locked_reference[newly_locked] = self.reference[newly_locked]
            command_position = quat_apply_inverse(
                self.robot.data.body_quat_w[:, self.mount_id], position - self.robot.data.body_pos_w[:, self.mount_id],
            )
            self.env.box_push_pose_override[newly_locked, :3] = command_position[newly_locked]
            self.env.box_push_pose_override[newly_locked, 3] = 1.0
            self.locked |= newly_locked
            self.locked &= ~at_goal
            self.motion_ready |= self.locked
        else:
            self.motion_ready |= self.env.box_push_approached
        jacobian = self.robot.root_physx_view.get_jacobians()[:, self.jacobian_body][:, :, self.jacobian_columns]
        desired = self.controller.compute(position, quaternion, jacobian, self.robot.data.joint_pos[:, self.joint_ids])
        limits = self.robot.data.soft_joint_pos_limits[:, self.joint_ids]
        desired = torch.clamp(desired, min=limits[..., 0], max=limits[..., 1])
        self.reference = coordinated_joint_target(self.reference, desired, self.env.step_dt)
        self.reference = torch.where(self.locked[:, None], self.locked_reference, self.reference)
        gravity = self.robot.root_physx_view.get_gravity_compensation_forces()
        columns = self.joint_ids if gravity.shape[1] == self.robot.num_joints else self.jacobian_columns
        compensated = self.reference + gravity[:, columns] / self.stiffness
        result = actions.clone()
        result[:, self.action_slots] = (compensated - self.robot.data.default_joint_pos[:, self.joint_ids]) / self.scale
        if self.hold_legs_until_ready:
            result[:, self.leg_slots] = torch.where(self.motion_ready[:, None], result[:, self.leg_slots], 0.0)
        return result