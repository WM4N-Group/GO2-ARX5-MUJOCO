"""Train a separate CLIMB actor to mount and settle on a low box."""

import isaaclab.sim as sim_utils
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass

from ... import mdp
from ...mdp import box_climb
from ...mdp.box_rewards import completion_reward
from ...mdp.box_terrain import single_box_terrain
from ...mdp.climb_start_states import reset_prepared_start
from .climb_env_cfg import ClimbRewardsCfg, Go2ARX5ClimbEnvCfg
from .box_robot_cfg import box_skill_robot_cfg


def feet_entities():
    return {
        "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
    }


@configclass
class SingleBoxTerrainCfg(SubTerrainBaseCfg):
    function = single_box_terrain
    box_height_range: tuple[float, float] = (0.05, 0.30)
    box_length: float = 1.2
    box_width: float = 1.2
    approach_distance: float = 0.75
    approach_height: float = 0.0
    approach_length: float = 1.2
    approach_width: float = 1.2
    approach_gap: float = 0.0
    approach_gap_range: tuple[float, float] | None = None
    following_platform_height: float = 0.0
    following_platform_length: float = 2.0
    following_platform_width: float = 1.6
    following_platform_gap: float = 0.02


@configclass
class BoxClimbRewardsCfg(ClimbRewardsCfg):
    completion = RewTerm(func=completion_reward, weight=50.0, params={"failure_terms": ("base_contact", "bad_orientation", "low_posture")})
    posture_failure = RewTerm(func=box_climb.posture_failure_cost, weight=-10.0)
    box_goal = RewTerm(func=box_climb.box_goal_reward, weight=2.0)
    box_support = RewTerm(func=box_climb.box_support_reward, weight=3.0, params=feet_entities())
    box_clearance = RewTerm(func=box_climb.box_clearance_reward, weight=2.0, params=feet_entities())
    box_standing = RewTerm(func=box_climb.box_standing, weight=5.0, params=feet_entities())
    flight = RewTerm(func=box_climb.feet_flight, weight=-0.5, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")})


@configclass
class Go2ARX5BoxClimbEnvCfg(Go2ARX5ClimbEnvCfg):
    rewards: BoxClimbRewardsCfg = BoxClimbRewardsCfg()
    prepared_start_path: str = ""
    prepared_start_phase: str = "ground"

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = box_skill_robot_cfg()
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.33)
        self.scene.num_envs = 2048
        self.episode_length_s = 12.0
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            seed=0, curriculum=True, size=(6.0, 6.0), border_width=4.0,
            num_rows=10, num_cols=8, use_cache=False,
            sub_terrains={"box": SingleBoxTerrainCfg(proportion=1.0)},
        )
        self.scene.terrain.max_init_terrain_level = 1
        self.scene.terrain.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="min", restitution_combine_mode="min",
            static_friction=0.8, dynamic_friction=0.6, restitution=0.0,
        )
        self.sim.physics_material = self.scene.terrain.physics_material
        self.observations.policy.velocity_commands.func = box_climb.box_velocity_commands
        self.observations.policy.velocity_commands.params = {}
        self.events.reset_base.params["pose_range"] = {"x": (-0.10, 0.10), "y": (-0.08, 0.08), "yaw": (-0.06, 0.06)}
        self.events.prepared_start = EventTerm(func=reset_prepared_start, mode="reset")
        self.rewards.track_lin_vel_xy_exp.func = box_climb.track_box_velocity
        self.rewards.track_lin_vel_xy_exp.params = {"std": 0.4}
        self.rewards.feet_air_time = None
        self.rewards.flat_orientation_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.15
        self.rewards.action_rate_l2.weight = -0.02
        self.rewards.dof_pos_limits.weight = -1.0
        self.terminations.box_settled = DoneTerm(func=box_climb.BoxSettled, params={**feet_entities(), "hold_time": 1.0})
        self.terminations.bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.9})
        self.terminations.low_posture = DoneTerm(func=box_climb.LowPosture, params=feet_entities())
        self.curriculum.terrain_levels = CurrTerm(func=box_climb.box_terrain_levels)


@configclass
class Go2ARX5BoxClimbEnvCfg_PLAY(Go2ARX5BoxClimbEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 4
        self.scene.terrain.terrain_generator.sub_terrains["box"].box_height_range = (0.25, 0.25)
        self.curriculum.terrain_levels = None