"""Low-stair curriculum for the GO2-ARX5 CLIMB skill."""

import isaaclab.terrains as terrain_gen
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg as RoughRewardsCfg,
)

from LeggedManip_Lab.assets.go2_arx5.go2_arx5_articulation_cfg import GO2_ARX5_CFG
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab import mdp

from ...leggedmanip_lab_env_cfg import ActionsCfg


CLIMB_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=0,
    curriculum=True,
    size=(8.0, 8.0),
    border_width=10.0,
    num_rows=8,
    num_cols=8,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "stairs": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0.02, 0.12),
            step_width=0.35,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    },
)


@configclass
class ClimbRewardsCfg(RoughRewardsCfg):
    arm_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint.*"]),
        },
    )


@configclass
class Go2ARX5ClimbEnvCfg(LocomotionVelocityRoughEnvCfg):
    actions: ActionsCfg = ActionsCfg()
    rewards: ClimbRewardsCfg = ClimbRewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.robot = GO2_ARX5_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        self.scene.height_scanner.offset.pos = (0.2, 0.0, 20.0)
        self.scene.terrain.terrain_generator = CLIMB_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = 1

        self.observations.policy.joint_pos.func = mdp.joint_pos_rel
        self.observations.policy.joint_vel.func = mdp.joint_vel_rel

        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.4, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        self.events.push_robot = None
        self.events.base_external_force_torque = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.base_com = None
        self.events.reset_robot_joints.params["position_range"] = (0.9, 1.1)
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.25, 0.25),
                "y": (-0.25, 0.25),
                "yaw": (-0.1, 0.1),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -0.5

        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"


@configclass
class Go2ARX5ClimbEnvCfg_PLAY(Go2ARX5ClimbEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.viewer.origin_type = "asset_body"
        self.viewer.env_index = 0
        self.viewer.asset_name = "robot"
        self.viewer.body_name = "base"
        self.viewer.eye = (3.0, 3.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.2)
        self.viewer.resolution = (960, 540)
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator.num_rows = 4
        self.scene.terrain.terrain_generator.num_cols = 4
        self.scene.terrain.terrain_generator.curriculum = False
        self.observations.policy.enable_corruption = False