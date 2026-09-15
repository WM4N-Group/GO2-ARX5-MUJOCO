"""PIPER-specific robot configurations for the existing physical box tasks."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from LeggedManip_Lab.assets.go2_piper.go2_piper_articulation_cfg import GO2_PIPER_CFG

from ..go2_arx5.box_climb_env_cfg import Go2ARX5BoxClimbEnvCfg
from ..go2_arx5.box_push_env_cfg import Go2ARX5BoxPushHybridEnvCfg
from ...mdp.box_climb import box_target_delta


PIPER_ARM_EFFORT_LIMITS = (20.0, 20.0, 15.0, 7.0, 5.0, 5.0)


def climb_target_distance(env):
    return box_target_delta(env)[:, :2].norm(dim=1)


def piper_box_robot_cfg():
    robot = GO2_PIPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos = (0.0, 0.0, 0.33)
    effort = {".*_hip_joint": 23.7, ".*_thigh_joint": 23.7, ".*_calf_joint": 45.43}
    velocity = {".*_hip_joint": 30.1, ".*_thigh_joint": 30.1, ".*_calf_joint": 15.7}
    legs = robot.actuators["base_legs"]
    legs.effort_limit = effort.copy()
    legs.effort_limit_sim = effort.copy()
    legs.velocity_limit = velocity.copy()
    legs.velocity_limit_sim = velocity.copy()
    for index, limit in enumerate(PIPER_ARM_EFFORT_LIMITS, 1):
        actuator = robot.actuators[f"joint{index}"]
        actuator.effort_limit = limit
        actuator.effort_limit_sim = limit
        actuator.velocity_limit = 3.0
        actuator.velocity_limit_sim = 3.0
    robot.spawn.articulation_props.solver_position_iteration_count = 8
    robot.spawn.articulation_props.solver_velocity_iteration_count = 2
    return robot


@configclass
class Go2PiperBoxPushHybridEnvCfg(Go2ARX5BoxPushHybridEnvCfg):
    box_size: tuple[float, float, float] = (1.2, 1.2, 0.20)

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = piper_box_robot_cfg()
        self.actions.joint_pos.lock_arm_on_contact = True
        self.actions.joint_pos.align_hand_orientation = True
        self.actions.joint_pos.hand_pitch = 0.8
        self.push_approach_distance = 0.03
        self.rewards.completion.weight = 150.0
        self.rewards.box_goal.weight = 0.0
        self.rewards.end_effector_position_tracking_exp.weight = 0.0
        self.rewards.track_base_height_exp.weight = 0.0
        self.rewards.track_ang_vel_z_exp.weight = 0.0
        self.rewards.hand_contact.weight = 0.0
        self.rewards.track_lin_vel_xy_exp.weight = 6.0


@configclass
class Go2PiperBoxPushHybridEnvCfg_PLAY(Go2PiperBoxPushHybridEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.box_mass = None
        self.events.box_material = None


@configclass
class Go2PiperBoxClimbEnvCfg(Go2ARX5BoxClimbEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = piper_box_robot_cfg()
        self.scene.terrain.terrain_generator.sub_terrains["box"].box_height_range = (0.05, 0.20)
        self.rewards.completion.weight = 200.0
        self.rewards.box_standing.weight = 1.0
        self.rewards.box_support.weight = 1.0
        self.rewards.box_clearance.weight = 0.5
        self.rewards.posture_failure.weight = -30.0
        self.rewards.box_center_distance = RewTerm(func=climb_target_distance, weight=-1.0)


@configclass
class Go2PiperBoxClimbEnvCfg_PLAY(Go2PiperBoxClimbEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 4
        self.scene.terrain.terrain_generator.sub_terrains["box"].box_height_range = (0.20, 0.20)
        self.curriculum.terrain_levels = None