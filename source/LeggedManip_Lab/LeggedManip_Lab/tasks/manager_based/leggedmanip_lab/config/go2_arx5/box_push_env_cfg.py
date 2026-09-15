"""Arm pushing of a finite-mass box with ordinary sliding friction."""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from ... import mdp
from ...leggedmanip_lab_env_cfg import RewardsCfg
from ...mdp import box_push
from ...mdp.box_rewards import completion_reward
from .box_robot_cfg import box_skill_robot_cfg
from .flat_env_cfg import Go2ARX5FlatEnvCfg


@configclass
class BoxObservationCfg(ObsGroup):
    state = ObsTerm(func=box_push.box_observation)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class BoxPushRewardsCfg(RewardsCfg):
    completion = RewTerm(func=completion_reward, weight=50.0, params={"failure_terms": ("base_contact", "bad_orientation", "box_contact", "box_tipped", "invalid_hand_contact")})
    box_progress = RewTerm(func=box_push.push_progress_reward, weight=8.0, params={"asset_cfg": SceneEntityCfg("robot", body_names="end_effector")})
    box_goal = RewTerm(func=box_push.push_goal_reward, weight=3.0)
    hand_contact = RewTerm(func=box_push.hand_contact, weight=1.0, params={"asset_cfg": SceneEntityCfg("robot", body_names="end_effector")})
    wrong_contact = RewTerm(func=box_push.forbidden_box_contact, weight=-10.0)


@configclass
class Go2ARX5BoxPushEnvCfg(Go2ARX5FlatEnvCfg):
    rewards: BoxPushRewardsCfg = BoxPushRewardsCfg()
    box_size: tuple[float, float, float] = (1.2, 1.2, 0.25)
    push_distance: float = 0.60
    push_press_depth: float = 0.02
    push_approach_distance: float = 0.10

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1024
        self.scene.env_spacing = 4.0
        self.episode_length_s = 12.0
        self.scene.robot = box_skill_robot_cfg()
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.33)
        self.scene.terrain.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="min", restitution_combine_mode="min",
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
        )
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.35, 0.4))
        self.scene.sky_light.spawn = sim_utils.DomeLightCfg(intensity=1500.0)
        self.scene.box = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Box",
            spawn=sim_utils.CuboidCfg(
                size=self.box_size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(linear_damping=0.0, angular_damping=0.0, max_depenetration_velocity=1.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="min", restitution_combine_mode="min",
                    static_friction=0.4, dynamic_friction=0.4, restitution=0.0,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.65, 0.3, 0.15)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(1.2, 0.0, self.box_size[2] / 2.0)),
        )
        self.scene.box_contact_hand = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/link6", filter_prim_paths_expr=["{ENV_REGEX_NS}/Box"],
            history_length=1, update_period=self.sim.dt, track_contact_points=True, max_contact_data_count_per_prim=32,
        )
        for name in box_push.NON_HAND_BODIES:
            setattr(self.scene, f"box_contact_{name}", ContactSensorCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{name}", filter_prim_paths_expr=["{ENV_REGEX_NS}/Box"],
                history_length=1, update_period=self.sim.dt,
            ))
        self.observations.box = BoxObservationCfg()
        self.observations.policy.velocity_commands.func = box_push.push_velocity_commands
        self.observations.policy.velocity_commands.params = {"asset_cfg": SceneEntityCfg("robot", body_names="end_effector")}
        self.observations.policy.pos_commands.func = box_push.push_ee_commands
        self.observations.policy.pos_commands.params = {"asset_cfg": SceneEntityCfg("robot", body_names="link0")}
        self.commands.base_velocity.debug_vis = False
        self.commands.ee_pose.debug_vis = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.events.add_mass = None
        self.events.base_com = None
        self.events.randomize_actuator_gains = None
        self.events.randomize_rigid_body_inertia = None
        self.events.base_external_force_torque = None
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "yaw": (-0.04, 0.04)},
            "velocity_range": {name: (0.0, 0.0) for name in ("x", "y", "z", "roll", "pitch", "yaw")},
        }
        self.events.reset_robot_joints.params["position_range"] = (0.95, 1.05)
        self.events.reset_box = EventTerm(func=box_push.reset_push_box, mode="reset")
        self.events.box_mass = EventTerm(func=mdp.randomize_rigid_body_mass, mode="startup", params={
            "asset_cfg": SceneEntityCfg("box"), "mass_distribution_params": (3.0, 8.0), "operation": "abs", "recompute_inertia": True,
        })
        self.events.box_material = EventTerm(func=mdp.randomize_rigid_body_material, mode="startup", params={
            "asset_cfg": SceneEntityCfg("box"), "static_friction_range": (0.2, 0.6),
            "dynamic_friction_range": (0.2, 0.6), "restitution_range": (0.0, 0.0), "num_buckets": 64, "make_consistent": True,
        })
        self.rewards.end_effector_position_tracking_exp.func = box_push.push_hand_tracking
        self.rewards.end_effector_position_tracking_exp.params = {"asset_cfg": SceneEntityCfg("robot", body_names="end_effector")}
        self.rewards.end_effector_orientation_tracking = None
        self.rewards.track_lin_vel_xy_exp.func = box_push.track_push_velocity
        self.rewards.track_lin_vel_xy_exp.params = {"std": 0.35, "asset_cfg": SceneEntityCfg("robot", body_names="end_effector")}
        self.rewards.track_ang_vel_z_exp.func = box_push.track_push_yaw
        self.rewards.track_ang_vel_z_exp.params = {"std": 0.5, "asset_cfg": SceneEntityCfg("robot", body_names="end_effector")}
        self.rewards.feet_air_time = None
        self.rewards.arm_deviation.weight = -0.02
        self.terminations.box_settled = DoneTerm(func=box_push.PushSettled, params={"asset_cfg": SceneEntityCfg("robot", body_names="end_effector"), "hold_time": 0.5})
        self.terminations.box_contact = DoneTerm(func=box_push.forbidden_box_contact)
        self.terminations.invalid_hand_contact = DoneTerm(func=box_push.invalid_hand_contact, params={"asset_cfg": SceneEntityCfg("robot", body_names="end_effector")})
        self.terminations.box_tipped = DoneTerm(func=box_push.box_tipped)
        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.ang_vel_cmd_levels = None
        self.curriculum.pos_cmd_levels = None


@configclass
class Go2ARX5BoxPushEnvCfg_PLAY(Go2ARX5BoxPushEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.box_mass = None
        self.events.box_material = None


@configclass
class Go2ARX5BoxPushHybridEnvCfg(Go2ARX5BoxPushEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.box.init_state.pos = (1.10, 0.0, self.box_size[2] / 2.0)
        from ...mdp.box_push_actions import BoxPushLegActionCfg, applied_joint_commands, hybrid_box_observation
        from ...mdp.box_climb import LowPosture
        from ...mdp.box_rewards import failure_cost

        failures = ("base_contact", "bad_orientation", "box_contact", "box_tipped", "invalid_hand_contact", "low_posture")
        self.rewards.completion.params["failure_terms"] = failures
        self.rewards.physical_failure = RewTerm(func=failure_cost, weight=-20.0, params={"failure_terms": failures})
        self.rewards.flat_orientation_l2.weight = -2.0
        self.terminations.low_posture = DoneTerm(func=LowPosture, params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        })
        original = self.actions.joint_pos
        self.actions.joint_pos = BoxPushLegActionCfg(
            asset_name=original.asset_name, joint_names=original.joint_names,
            scale=original.scale, use_default_offset=True, preserve_order=True, clip=original.clip,
        )
        self.observations.policy.actions.func = applied_joint_commands
        self.observations.critic.actions.func = applied_joint_commands
        self.observations.box.state.func = hybrid_box_observation
        self.observations.box.state.params = {"asset_cfg": SceneEntityCfg("robot", body_names="end_effector")}


@configclass
class Go2ARX5BoxPushHybridEnvCfg_PLAY(Go2ARX5BoxPushHybridEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.box_mass = None
        self.events.box_material = None