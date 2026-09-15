"""Explicit actuator limits for the new box-skill training profile."""

from LeggedManip_Lab.assets.go2_arx5.go2_arx5_articulation_cfg import GO2_ARX5_CFG


def box_skill_robot_cfg():
    robot = GO2_ARX5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    effort = {".*_hip_joint": 23.7, ".*_thigh_joint": 23.7, ".*_calf_joint": 45.43}
    velocity = {".*_hip_joint": 30.1, ".*_thigh_joint": 30.1, ".*_calf_joint": 15.7}
    legs = robot.actuators["base_legs"]
    legs.effort_limit = effort.copy()
    legs.effort_limit_sim = effort.copy()
    legs.velocity_limit = velocity.copy()
    legs.velocity_limit_sim = velocity.copy()
    for index, limit in enumerate((20.0, 20.0, 20.0, 7.0, 5.0, 5.0), 1):
        arm = robot.actuators[f"joint{index}"]
        arm.effort_limit = limit
        arm.effort_limit_sim = limit
        arm.velocity_limit = 3.0
        arm.velocity_limit_sim = 3.0
    robot.spawn.articulation_props.solver_position_iteration_count = 8
    robot.spawn.articulation_props.solver_velocity_iteration_count = 2
    return robot