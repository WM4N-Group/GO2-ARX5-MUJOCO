"""Train leg actions while a constrained arm controller handles contact."""

import torch

from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.managers import ActionTerm, SceneEntityCfg
from isaaclab.utils import configclass

from .box_push import box_observation, hand_contact
from .box_push_ik import PushArmIK


class BoxPushLegAction(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.delegate = JointPositionAction(cfg, env)
        self.controller = PushArmIK(env, hold_legs_until_ready=True, lock_arm_on_contact=cfg.lock_arm_on_contact, align_hand_orientation=cfg.align_hand_orientation, hand_pitch=cfg.hand_pitch)
        self._raw = torch.zeros((env.num_envs, len(self.controller.leg_slots)), device=env.device)
        self.joint_commands = torch.zeros((env.num_envs, len(self.controller.leg_slots) + len(self.controller.action_slots)), device=env.device)

    @property
    def action_dim(self):
        return len(self.controller.leg_slots)

    @property
    def raw_actions(self):
        return self._raw

    @property
    def processed_actions(self):
        return self.delegate.processed_actions

    def process_actions(self, actions):
        self._raw[:] = actions
        combined = torch.zeros_like(self.joint_commands)
        combined[:, self.controller.leg_slots] = actions
        self.joint_commands[:] = self.controller.apply(combined)
        self.delegate.process_actions(self.joint_commands)

    def apply_actions(self):
        self.delegate.apply_actions()

    def reset(self, env_ids=None):
        selected = slice(None) if env_ids is None else env_ids
        self._raw[selected] = 0.0
        self.joint_commands[selected] = 0.0
        self.delegate.reset(env_ids)
        self.controller.reset(env_ids)


@configclass
class BoxPushLegActionCfg(JointPositionActionCfg):
    class_type = BoxPushLegAction
    lock_arm_on_contact: bool = True
    align_hand_orientation: bool = False
    hand_pitch: float = 0.0


def applied_joint_commands(env):
    return env.action_manager.get_term("joint_pos").joint_commands


def hybrid_box_observation(env, asset_cfg=SceneEntityCfg("robot", body_names="end_effector")):
    return torch.cat((
        box_observation(env), env.box_push_pose_override_active[:, None].float(),
        hand_contact(env, asset_cfg)[:, None].float(),
    ), dim=1)