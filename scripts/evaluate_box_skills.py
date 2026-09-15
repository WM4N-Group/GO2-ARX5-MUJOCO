"""Deterministic physical evaluation of the separate box-skill actors."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import runpy
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True, choices=("GO2-ARX5-Box-Climb-Play", "GO2-ARX5-Box-Push-Play", "GO2-ARX5-Box-Push-Hybrid-Play", "GO2-PIPER-Box-Climb-Play", "GO2-PIPER-Box-Push-Hybrid-Play"))
source_group = parser.add_mutually_exclusive_group(required=True)
source_group.add_argument("--checkpoint", type=Path)
source_group.add_argument("--actor-policy", type=Path, help="Diagnostic: directly evaluate a provenance-recorded TorchScript skill actor.")
source_group.add_argument("--locomotion-policy", type=Path, help="Use the frozen 210-input NAV actor for legs; requires --arm-ik.")
parser.add_argument("--output-json", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=1200)
parser.add_argument("--episodes-per-env", type=int, default=1)
parser.add_argument("--episode-length-s", type=float)
parser.add_argument("--preparation-seconds", type=float, default=0.0)
parser.add_argument("--seed", type=int, default=100)
parser.add_argument("--box-height", type=float)
parser.add_argument("--approach-height", type=float, default=0.0)
parser.add_argument("--approach-gap", type=float, default=0.0)
parser.add_argument("--landing-length", type=float)
parser.add_argument("--landing-width", type=float)
parser.add_argument("--start-clearance", type=float)
parser.add_argument("--prepared-starts", type=Path)
parser.add_argument("--prepared-phase", choices=("ground", "platform"), default="ground")
parser.add_argument("--following-platform-height", type=float, default=0.0)
parser.add_argument("--surface-friction", type=float)
parser.add_argument("--box-mass", type=float)
parser.add_argument("--box-friction", type=float)
parser.add_argument("--press-depth", type=float)
parser.add_argument("--push-distance", type=float)
parser.add_argument("--arm-ik", action="store_true", help="Diagnostic: replace the learned arm output with rate-limited differential IK.")
parser.add_argument("--neutral-arm", action="store_true", help="Diagnostic: hold the nominal arm pose for CLIMB.")
parser.add_argument("--neutral-push-command", action="store_true", help="Diagnostic: keep the NAV actor end-effector command at its neutral reference during PUSH.")
parser.add_argument("--observation-noise", action="store_true", help="Diagnostic: retain training observation noise during evaluation.")
parser.add_argument("--video-path", type=Path)
parser.add_argument("--deployment-actor", type=Path, help="Compare exported TorchScript outputs on every evaluated observation.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.arm_ik and "Push" not in args.task:
    parser.error("--arm-ik only applies to PUSH")
if args.neutral_arm and "Climb" not in args.task:
    parser.error("--neutral-arm only applies to CLIMB")
if args.neutral_push_command and "Push" not in args.task:
    parser.error("--neutral-push-command only applies to PUSH")
if args.arm_ik and "Hybrid" in args.task:
    parser.error("The hybrid task already contains its arm controller")
if args.locomotion_policy is not None and not args.arm_ik:
    parser.error("--locomotion-policy requires --arm-ik")
if args.output_json.exists():
    parser.error("Output file already exists")
if args.video_path is not None:
    if args.video_path.exists():
        parser.error("Video file already exists")
    args.enable_cameras = True
if args.steps < 1 or args.num_envs < 1 or args.episodes_per_env < 1:
    parser.error("Evaluation size must be positive")
if not math.isfinite(args.preparation_seconds) or args.preparation_seconds < 0.0:
    parser.error("Preparation duration must be finite and nonnegative")
if args.preparation_seconds and ("Climb" not in args.task or args.episodes_per_env != 1):
    parser.error("Explicit preparation currently supports one CLIMB episode per environment")
for value in (args.box_height, args.box_mass, args.box_friction, args.episode_length_s, args.landing_length, args.landing_width, args.start_clearance, args.surface_friction):
    if value is not None and (not math.isfinite(value) or value <= 0.0):
        parser.error("Physical parameters must be positive and finite")
for value in (args.approach_height, args.approach_gap, args.following_platform_height):
    if not math.isfinite(value) or value < 0.0:
        parser.error("Approach height and gap must be finite and nonnegative")
for value in (args.press_depth, args.push_distance):
    if value is not None and (not math.isfinite(value) or value <= 0.0):
        parser.error("PUSH parameters must be positive and finite")
launcher = AppLauncher(args)

import gymnasium as gym
import torch

import isaaclab_tasks
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import LeggedManip_Lab.tasks
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.mdp import box_climb, box_push
from box_evaluation_metrics import EpisodeQuota


def main():
    config = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    config.scene.num_envs = args.num_envs
    config.seed = args.seed
    config.sim.device = args.device
    config.observations.policy.enable_corruption = args.observation_noise
    if args.episode_length_s is not None:
        config.episode_length_s = args.episode_length_s
    climbing = "Climb" in args.task
    if args.prepared_starts is not None:
        if not climbing:
            raise ValueError("Prepared starts require the CLIMB task")
        config.prepared_start_path = str(args.prepared_starts)
        config.prepared_start_phase = args.prepared_phase
    if args.start_clearance is not None:
        config.scene.robot.init_state.pos = (*config.scene.robot.init_state.pos[:2], args.start_clearance)
    if climbing:
        terrain = config.scene.terrain.terrain_generator.sub_terrains["box"]
        terrain.approach_height = args.approach_height
        terrain.approach_gap = args.approach_gap
        terrain.following_platform_height = args.following_platform_height
        if args.surface_friction is not None:
            config.scene.terrain.physics_material.static_friction = args.surface_friction
            config.scene.terrain.physics_material.dynamic_friction = args.surface_friction
            config.sim.physics_material = config.scene.terrain.physics_material
        if args.landing_length is not None:
            terrain.box_length = args.landing_length
        if args.landing_width is not None:
            terrain.box_width = args.landing_width
    if args.box_height is not None:
        if climbing:
            config.scene.terrain.terrain_generator.sub_terrains["box"].box_height_range = (args.box_height, args.box_height)
        else:
            config.box_size = (*config.box_size[:2], args.box_height)
            config.scene.box.spawn.size = config.box_size
            config.scene.box.init_state.pos = (*config.scene.box.init_state.pos[:2], args.box_height / 2.0)
    if not climbing:
        if args.press_depth is not None:
            config.push_press_depth = args.press_depth
        if args.push_distance is not None:
            config.push_distance = args.push_distance
        if args.box_mass is not None:
            config.scene.box.spawn.mass_props.mass = args.box_mass
        if args.box_friction is not None:
            config.scene.box.spawn.physics_material.static_friction = args.box_friction
            config.scene.box.spawn.physics_material.dynamic_friction = args.box_friction
    agent_config = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_config.device = args.device
    agent_config.seed = args.seed
    if agent_config.actor.activation != "elu":
        raise ValueError("Evaluation requires the project's ELU actor")
    env = RslRlVecEnvWrapper(gym.make(args.task, cfg=config, render_mode="rgb_array" if args.video_path else None), clip_actions=agent_config.clip_actions)
    base = env.unwrapped
    video = None
    if args.video_path is not None:
        import imageio.v2 as imageio

        args.video_path.parent.mkdir(parents=True, exist_ok=True)
        video = imageio.get_writer(str(args.video_path), fps=round(1.0 / (2 * base.step_dt)), codec="libx264")
        origin = base.scene.env_origins[0].cpu().numpy()
        camera_eye = [3.0, 3.0, 1.8] if climbing else [0.7, 3.2, 1.5]
        base.sim.set_camera_view(eye=origin + camera_eye, target=origin + [0.9, 0.0, 0.3])
    if args.locomotion_policy is not None:
        policy = torch.jit.load(str(args.locomotion_policy), map_location=args.device).eval()
        observation_groups = ["policy"]
    elif args.actor_policy is not None:
        policy = torch.jit.load(str(args.actor_policy), map_location=args.device).eval()
        observation_groups = agent_config.obs_groups["actor"]
    else:
        checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        load_actor = runpy.run_path(str(Path(__file__).resolve().parent / "rsl_rl/initialization.py"))["actor_from_state_dict"]
        policy = load_actor(checkpoint["actor_state_dict"], device=args.device)
        observation_groups = agent_config.obs_groups["actor"]
    observation = env.get_observations()
    deployment_actor = torch.jit.load(str(args.deployment_actor), map_location=args.device).eval() if args.deployment_actor else None
    deployment_max_error = 0.0
    deployment_comparisons = 0
    robot = base.scene["robot"]
    arm_controller = None
    if args.arm_ik:
        from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.mdp.box_push_ik import PushArmIK

        arm_controller = PushArmIK(base, hold_legs_until_ready=args.locomotion_policy is not None, lock_arm_on_contact=args.locomotion_policy is not None)
    feet = SceneEntityCfg("robot", body_names=".*_foot")
    feet.resolve(base.scene)
    sensor = SceneEntityCfg("contact_forces", body_names=".*_foot")
    sensor.resolve(base.scene)
    hand = SceneEntityCfg("robot", body_names="end_effector")
    hand.resolve(base.scene)
    arm = SceneEntityCfg("robot", joint_names="joint.*")
    arm.resolve(base.scene)
    quota = EpisodeQuota(args.num_envs, args.episodes_per_env, args.device)
    completions, successes = quota.completed, quota.successful
    terminal_states = []
    original_pre_reset = base.recorder_manager.record_pre_reset

    def record_pre_reset(env_ids, *arguments, **keywords):
        fields = {
            "elapsed_seconds": base.episode_length_buf * base.step_dt,
            "robot_position": robot.data.root_pos_w - base.scene.env_origins,
            "robot_velocity": robot.data.root_lin_vel_w,
            "robot_gravity": robot.data.projected_gravity_b,
            "termination_flags": torch.stack([
                base.termination_manager.get_term(name) for name in base.termination_manager.active_terms
            ], dim=1),
        }
        if not climbing:
            box = base.scene["box"]
            target = box_push.push_target(base)
            displacement = box.data.root_pos_w - target
            displacement = displacement.clone()
            displacement[:, 0] += config.push_distance
            terminal_hand_force = base.scene["box_contact_hand"].data.force_matrix_w.sum(dim=(1, 2))
            terminal_contact = base.scene["box_contact_hand"].data.contact_pos_w[:, 0, 0]
            terminal_valid = box_push.hand_contact(base, hand)
            terminal_strength = torch.linalg.vector_norm(terminal_hand_force, dim=1)
            selected = env_ids[quota.active[env_ids]]
            unclassified = (terminal_strength > 1.0) & ~terminal_valid
            unclassified_hand_contact_steps[selected] += unclassified[selected].float()
            maximum_unclassified_hand_force[selected] = torch.maximum(
                maximum_unclassified_hand_force[selected], terminal_strength[selected] * unclassified[selected],
            )
            fields.update(
                box_position=box.data.root_pos_w - base.scene.env_origins,
                box_quaternion=box.data.root_quat_w,
                box_goal=target - base.scene.env_origins,
                box_displacement=displacement,
                box_goal_error=torch.linalg.vector_norm(target[:, :2] - box.data.root_pos_w[:, :2], dim=1),
                box_speed=torch.linalg.vector_norm(box.data.root_lin_vel_w, dim=1),
                box_angular_speed=torch.linalg.vector_norm(box.data.root_ang_vel_w, dim=1),
                hand_force=terminal_hand_force,
                hand_position=robot.data.body_pos_w[:, hand.body_ids[0]] - base.scene.env_origins,
                hand_contact_position=torch.nan_to_num(terminal_contact - base.scene.env_origins),
                hand_contact_position_valid=torch.isfinite(terminal_contact).all(dim=1),
                hand_contact_valid=terminal_valid,
            )
        terminal_states.extend(quota.terminal_snapshot(env_ids, **fields))
        return original_pre_reset(env_ids, *arguments, **keywords)

    base.recorder_manager.record_pre_reset = record_pre_reset
    maximum_forward = torch.full((args.num_envs,), -torch.inf, device=args.device)
    minimum_base_height = torch.full((args.num_envs,), torch.inf, device=args.device)
    maximum_tilt = torch.zeros(args.num_envs, device=args.device)
    maximum_condition = torch.zeros(args.num_envs, device=args.device)
    condition_duration = torch.zeros_like(maximum_condition)
    maximum_support = torch.zeros_like(maximum_condition)
    maximum_joint_velocity = torch.zeros(robot.num_joints, device=args.device)
    maximum_torque_ratio = 0.0
    push_force_sum = torch.zeros(args.num_envs, device=args.device)
    push_contact_steps = torch.zeros_like(push_force_sum)
    maximum_push_force = torch.zeros_like(push_force_sum)
    unclassified_hand_contact_steps = torch.zeros_like(push_force_sum)
    maximum_unclassified_hand_force = torch.zeros_like(push_force_sum)
    reasons = {name: 0 for name in base.termination_manager.active_terms}
    trace = []
    with torch.inference_mode():
        for step in range(args.steps):
            active = quota.active.clone()
            relative = robot.data.root_pos_w - base.scene.env_origins
            maximum_forward[active] = torch.maximum(maximum_forward[active], relative[active, 0])
            minimum_base_height[active] = torch.minimum(minimum_base_height[active], relative[active, 2])
            tilt = torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0))
            maximum_tilt[active] = torch.maximum(maximum_tilt[active], tilt[active])
            if climbing:
                condition = box_climb.box_standing(base, feet, sensor)
                support = box_climb.supported_feet(base, feet, sensor).sum(dim=1).float()
                maximum_support[active] = torch.maximum(maximum_support[active], support[active])
            else:
                condition = box_push.hand_contact(base, hand)
                hand_force = base.scene["box_contact_hand"].data.force_matrix_w.sum(dim=(1, 2))
                normal_force = torch.linalg.vector_norm(hand_force, dim=1)
                push_force_sum += normal_force * (condition & active)
                push_contact_steps += (condition & active).float()
                maximum_push_force[active] = torch.maximum(maximum_push_force[active], normal_force[active])
                unclassified = (normal_force > 1.0) & ~condition & active
                unclassified_hand_contact_steps += unclassified.float()
                maximum_unclassified_hand_force = torch.maximum(maximum_unclassified_hand_force, normal_force * unclassified)
            condition_duration = torch.where(condition & active, condition_duration + base.step_dt, 0.0)
            maximum_condition = torch.maximum(maximum_condition, condition_duration)
            maximum_joint_velocity = torch.maximum(maximum_joint_velocity, robot.data.joint_vel[active].abs().max(dim=0).values)
            ratio = robot.data.applied_torque[active].abs() / robot.data.joint_effort_limits[active].clamp_min(1e-6)
            maximum_torque_ratio = max(maximum_torque_ratio, ratio.max().item())
            if step % 25 == 0 and active[0]:
                sample = {
                    "step": step, "robot_position": relative[0].tolist(),
                    "robot_gravity": robot.data.projected_gravity_b[0].tolist(),
                    "robot_velocity": robot.data.root_lin_vel_w[0].tolist(),
                    "feet_height": (robot.data.body_pos_w[0, feet.body_ids, 2] - base.scene.env_origins[0, 2]).tolist(),
                }
                if not climbing:
                    target = box_push.push_hand_target(base)[0]
                    sample.update(
                        box_position=(base.scene["box"].data.root_pos_w[0] - base.scene.env_origins[0]).tolist(),
                        hand_position=(robot.data.body_pos_w[0, hand.body_ids[0]] - base.scene.env_origins[0]).tolist(),
                        hand_contact=bool(condition[0]),
                        approach_completed=bool(base.box_push_approached[0]),
                        hand_force_world=hand_force[0].tolist(),
                        hand_target_error=(target - robot.data.body_pos_w[0, hand.body_ids[0]]).tolist(),
                        velocity_command=box_push.push_velocity_commands(base, hand)[0].tolist(),
                        arm_torque_limit_ratio=(robot.data.applied_torque[0, arm.joint_ids].abs() / robot.data.joint_effort_limits[0, arm.joint_ids]).tolist(),
                    )
                trace.append(sample)
            if video is not None and step % 2 == 0:
                video.append_data(base.render())
            actor_input = torch.cat([observation[name] for name in observation_groups], dim=-1)
            if args.neutral_push_command:
                actor_input[:, 189:210] = actor_input.new_tensor((0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0)).repeat(3)
            actions = policy(actor_input)
            if deployment_actor is not None:
                deployed_actions = deployment_actor(actor_input)
                torch.testing.assert_close(deployed_actions, actions, rtol=1e-6, atol=1e-6)
                deployment_max_error = max(deployment_max_error, (deployed_actions - actions).abs().max().item())
                deployment_comparisons += int(active.sum().item())
            if arm_controller is not None:
                actions = arm_controller.apply(actions)
            if args.neutral_arm:
                actions = actions.clone()
                actions[:, 12:] = 0.0
            if step * base.step_dt < args.preparation_seconds:
                actions = torch.zeros_like(actions)
            observation, _rewards, dones, _extras = env.step(actions)
            done = dones.bool()
            if arm_controller is not None:
                arm_controller.reset(done.nonzero(as_tuple=False).flatten())
            success = done & base.termination_manager.get_term("box_settled")
            for name in base.termination_manager.active_terms:
                if name not in ("box_settled", "time_out"):
                    success &= ~base.termination_manager.get_term(name)
            counted = quota.record(done, success)
            for name in reasons:
                reasons[name] += int((counted & base.termination_manager.get_term(name)).sum().item())
            condition_duration[done] = 0.0
            if (step + 1) % 200 == 0:
                print(f"EVAL_STEP={step + 1} completed={completions.sum().item()} successes={successes.sum().item()}", flush=True)
            if not quota.active.any():
                break
    if video is not None:
        video.close()
    if climbing:
        terrain = config.scene.terrain.terrain_generator.sub_terrains["box"]
        actual_physics = {
            "box_height_range": terrain.box_height_range, "fixed_support": True,
            "approach_height": terrain.approach_height, "approach_gap": terrain.approach_gap,
            "approach_gap_range": terrain.approach_gap_range,
            "approach_length": terrain.approach_length, "approach_width": terrain.approach_width,
            "landing_length": terrain.box_length, "landing_width": terrain.box_width,
            "following_platform_height": terrain.following_platform_height,
            "following_platform_gap": terrain.following_platform_gap,
            "terrain_static_friction": config.scene.terrain.physics_material.static_friction,
            "terrain_dynamic_friction": config.scene.terrain.physics_material.dynamic_friction,
            "start_clearance": config.scene.robot.init_state.pos[2],
            "prepared_start_path": config.prepared_start_path,
            "prepared_start_phase": config.prepared_start_phase if config.prepared_start_path else None,
            "prepared_start_sha256": hashlib.sha256(Path(config.prepared_start_path).read_bytes()).hexdigest() if config.prepared_start_path else None,
        }
    else:
        box = base.scene["box"]
        actual_physics = {
            "box_mass_per_env": box.root_physx_view.get_masses().flatten().tolist(),
            "box_material_per_env": box.root_physx_view.get_material_properties().tolist(),
            "box_size": config.box_size, "fixed_support": False,
            "push_press_depth": config.push_press_depth, "push_distance": config.push_distance,
            "push_approach_distance": config.push_approach_distance,
        }
    mdp_directory = Path(box_push.__file__).resolve().parent
    repository = Path(__file__).resolve().parents[1]
    robot_asset_directory = Path(config.scene.robot.spawn.usd_path).resolve().parent
    robot_asset_paths = sorted(robot_asset_directory.rglob("*.usd"))
    robot_asset_paths.extend(robot_asset_directory.glob("*.py"))
    actual_physics.update(
        robot_asset_sha256={str(path.relative_to(repository)): hashlib.sha256(path.read_bytes()).hexdigest() for path in robot_asset_paths},
        robot_joint_names=list(robot.joint_names),
        robot_effort_limits=robot.data.joint_effort_limits[0].tolist(),
        robot_joint_defaults=robot.data.default_joint_pos[0].tolist(),
    )
    if "Hybrid" in args.task:
        controller = base.action_manager.get_term("joint_pos").controller
        actual_physics["controller_contract"] = {
            "lock_arm_on_contact": controller.lock_arm_on_contact,
            "align_hand_orientation": getattr(controller, "align_hand_orientation", False),
            "hand_pitch": getattr(config.actions.joint_pos, "hand_pitch", 0.0),
            "approach_distance": config.push_approach_distance,
            "press_depth": config.push_press_depth, "control_dt": base.step_dt,
        }
    control_files = ("box_climb.py", "box_rewards.py", "box_terrain.py") if climbing else ("box_push.py", "box_push_control.py", "box_rewards.py")
    if climbing and config.prepared_start_path:
        control_files += ("climb_start_states.py",)
    if args.arm_ik:
        control_files += ("box_push_ik.py",)
    if "Hybrid" in args.task:
        control_files += ("box_push_ik.py", "box_push_actions.py")
    result = {
        "schema_version": 4, "episodes_per_env": args.episodes_per_env,
        "neutral_arm_diagnostic": args.neutral_arm,
        "neutral_push_command_diagnostic": args.neutral_push_command,
        "preparation_seconds": args.preparation_seconds,
        "observation_noise_diagnostic": args.observation_noise,
        "terminal_states": terminal_states,
        "termination_flag_names": list(base.termination_manager.active_terms),
        "requested_episodes": args.num_envs * args.episodes_per_env,
        "incomplete_episodes": int((args.episodes_per_env - completions).sum().item()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None,
        "actor_policy": str(args.actor_policy) if args.actor_policy else None,
        "actor_policy_sha256": hashlib.sha256(args.actor_policy.read_bytes()).hexdigest() if args.actor_policy else None,
        "locomotion_policy": str(args.locomotion_policy) if args.locomotion_policy else None,
        "locomotion_policy_sha256": hashlib.sha256(args.locomotion_policy.read_bytes()).hexdigest() if args.locomotion_policy else None,
        "evaluation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "deployment_actor_sha256": hashlib.sha256(args.deployment_actor.read_bytes()).hexdigest() if args.deployment_actor else None,
        "deployment_actor_max_abs_error": deployment_max_error if deployment_actor is not None else None,
        "deployment_actor_comparisons": deployment_comparisons,
        "control_sha256": {name: hashlib.sha256((mdp_directory / name).read_bytes()).hexdigest() for name in control_files},
        "actual_physics": actual_physics, "executed_steps": step + 1,
        "episode_length_s": config.episode_length_s,
        "arm_controller": "trained_legs_with_ik_arm" if "Hybrid" in args.task else "differential_ik_gravity_diagnostic" if args.arm_ik else "learned_policy",
        "stand_until_arm_ready": args.locomotion_policy is not None or "Hybrid" in args.task,
        "lock_arm_on_contact": base.action_manager.get_term("joint_pos").controller.lock_arm_on_contact if "Hybrid" in args.task else args.locomotion_policy is not None,
        "trace_env0": trace, "video_path": str(args.video_path) if args.video_path else None,
        "task": args.task, "checkpoint": str(args.checkpoint) if args.checkpoint else None, "seed": args.seed,
        "num_envs": args.num_envs, "steps": args.steps, "box_height": args.box_height,
        "box_mass": args.box_mass, "box_friction": args.box_friction,
        "completed_episodes": int(completions.sum().item()), "successful_episodes": int(successes.sum().item()),
        "per_env_completions": completions.tolist(), "per_env_successes": successes.tolist(),
        "termination_reasons": reasons, "max_forward_per_env": maximum_forward.tolist(),
        "min_base_height_per_env": minimum_base_height.tolist(), "max_tilt_radians_per_env": maximum_tilt.tolist(),
        "max_standing_seconds" if climbing else "max_hand_contact_seconds": maximum_condition.tolist(),
        "max_supported_feet": maximum_support.tolist() if climbing else None,
        "max_torque_limit_ratio": maximum_torque_ratio,
        "push_contact_steps": push_contact_steps.tolist() if not climbing else None,
        "unclassified_hand_contact_steps": unclassified_hand_contact_steps.tolist() if not climbing else None,
        "max_unclassified_hand_force": maximum_unclassified_hand_force.tolist() if not climbing else None,
        "mean_hand_force_on_contact": (push_force_sum / push_contact_steps.clamp_min(1.0)).tolist() if not climbing else None,
        "max_hand_force": maximum_push_force.tolist() if not climbing else None,
        "joint_names": robot.joint_names, "max_abs_joint_velocity": maximum_joint_velocity.tolist(),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print("BOX_SKILL_EVALUATION=" + json.dumps(result, allow_nan=False))
    env.close()


try:
    main()
except Exception:
    traceback.print_exc()
    try:
        launcher.app.close()
    finally:
        raise SystemExit(1)
else:
    launcher.app.close()