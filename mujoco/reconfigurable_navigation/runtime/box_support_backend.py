"""Frozen box controllers behind the shared production executor interface."""

import numpy as np

from ..box_climb_runtime import BoxClimbRuntime
from ..box_navigation_runtime import BoxNavigationRuntime
from ..box_support_control import clear_arm, climb_surface, position_for_climb, prepare_climb, push_box
from ..box_support_env import BOX_ID, PLATFORM_ID, BoxSupportEnv, make_episode
from ..box_support_planner import BoxSupportPlanner, operation_key
from ..representations import SkillType
from ..skills import ClimbSkill, NavigateSkill, PushSkill, SkillStatus
from ..skills.climb import ClimbConfig
from ..skills.navigate import NavigateConfig
from ..skills.push import PushConfig, PushPhase
from .executor import ReconfigurableExecutor
from .replanner import OracleReplanner
from .safety import SafetyConfig, SafetyMonitor


class _ExecutionInterrupted(Exception):
    pass


class BoxSupportBackend:
    def __init__(self, env, climb_policy, platform_policy):
        self.env = env
        self.push = env.push_runtime
        self.climb = BoxClimbRuntime(climb_policy, model=env.model, data=env.data)
        self.platform = BoxClimbRuntime(platform_policy, model=env.model, data=env.data)
        for runtime in (self.climb, self.platform):
            runtime.joint_velocity_limits[12:] = 3.0
        self.active_physics = self.push
        self.completed = set()
        self.primitive_reports = []
        self.stage = "execution"
        self.phase = None

    def snapshot_runtimes(self):
        return tuple(runtime for runtime in (self.push, self.climb, self.platform) if runtime is not None)

    def initial_stage(self, action):
        return "preparation" if action.skill in (SkillType.PUSH, SkillType.CLIMB) else "execution"

    def create_skill(self, action):
        objects = {obj.object_id: obj for obj in self.env.observe().objects}
        if action.skill == SkillType.PUSH and action.object_id == BOX_ID:
            expected = self.env.push_target
            skill = PushSkill(PushConfig(timeout_steps=800))
        elif action.skill in (SkillType.NAV, SkillType.CLIMB) and action.support_id in objects:
            obj = objects[action.support_id]
            if action.skill == SkillType.NAV:
                expected = obj.climb_entry_pose
                skill = NavigateSkill(NavigateConfig(timeout_steps=400))
            else:
                expected = obj.climb_landing_pose
                skill = ClimbSkill(ClimbConfig(timeout_steps=800, stable_steps=50))
        else:
            return None
        if action.target_pose.shape != expected.shape or not np.allclose(action.target_pose, expected, rtol=0.0, atol=1e-5):
            return None
        return skill

    def execute_skill(self, action, skill, after_step):
        self.env.active_skill = action.skill
        steps = 0
        listeners = []

        def bind(runtime):
            previous = getattr(runtime, "on_step", None)

            def control_step(current):
                nonlocal steps
                stage, phase = self.stage, self.phase
                if isinstance(skill, PushSkill) and self.phase != "clear_arm":
                    skill.phase = PushPhase.VERIFY if self.push.arm.at_goal() else PushPhase.PUSH if self.push.arm.motion_ready else PushPhase.CONTACT
                    stage = "execution" if self.push.arm.motion_ready else "preparation"
                if isinstance(skill, PushSkill):
                    phase = skill.phase.value
                steps += int(stage == "execution")
                skill.steps = steps
                if previous is not None:
                    previous(current)
                if not after_step(stage, phase):
                    raise _ExecutionInterrupted()

            runtime.on_step = control_step
            listeners.append((runtime, previous))
            self.active_physics = runtime

        def perform(name, stage, function, *args, **kwargs):
            self.stage, self.phase = stage, name.lower()
            result = function(*args, **kwargs)
            self.primitive_reports.append({**result, "skill": name})
            if not result["succeeded"]:
                skill.failure_reason = result["reason"]
            return result["succeeded"]

        try:
            if action.skill == SkillType.PUSH:
                self.push.arm.goal = self.env.push_goal.copy()
                self.push.activate()
                bind(self.push)
                success = perform("PUSH", "execution", push_box, self.push, self.env.seed)
                if success:
                    skill.phase = PushPhase.RETREAT
                    success = perform("CLEAR_ARM", "execution", clear_arm, self.push)
            else:
                if action.skill == SkillType.NAV:
                    if self.active_physics is not self.climb:
                        self.climb.inherit_actuator_state(self.active_physics)
                    bind(self.climb)
                    raised = action.support_id == PLATFORM_ID
                    success = perform("POSITION_PLATFORM" if raised else "POSITION_CLIMB", "execution", position_for_climb, self.climb, self.env.object_geoms[action.support_id], support_height=0.20 if raised else 0.0, blend_seconds=0.5 if raised else 0.0)
                else:
                    raised = action.support_id == PLATFORM_ID
                    if raised:
                        self.platform.activate()
                    runtime = self.platform if raised else self.climb
                    if runtime is not self.active_physics:
                        runtime.inherit_actuator_state(self.active_physics)
                    bind(runtime)
                    success = perform("PREPARE_PLATFORM" if raised else "PREPARE_CLIMB", "preparation", prepare_climb, runtime, self.push, support_surface=self.env.box_geom_id if raised else None)
                    if success:
                        success = perform("CLIMB_PLATFORM" if raised else "CLIMB_BOX", "execution", climb_surface, runtime, self.env.object_geoms[action.support_id], self.env.box_geom_id)
            skill.status = SkillStatus.SUCCEEDED if success else SkillStatus.FAILED
            if success:
                self.completed.add(operation_key(action))
            return skill.status, steps, False
        except _ExecutionInterrupted:
            skill.status = SkillStatus.FAILED
            skill.failure_reason = None
            return skill.status, steps, True
        finally:
            for runtime, previous in listeners:
                runtime.on_step = previous


def make_box_support_executor(push_policy, climb_policy, platform_policy, seed):
    push, _start = make_episode(push_policy, seed, platform_height=0.40)
    env = BoxSupportEnv(push, seed)
    backend = BoxSupportBackend(env, climb_policy, platform_policy)
    navigation = BoxNavigationRuntime(push)
    executor = ReconfigurableExecutor(
        env, navigation,
        replanner=OracleReplanner(env.push_target, BoxSupportPlanner(backend.completed)),
        safety=SafetyMonitor(SafetyConfig(max_skill_failures=1)),
        skill_backend=backend, stabilization_duration=0.0, terminal_hold=False,
    )
    return executor