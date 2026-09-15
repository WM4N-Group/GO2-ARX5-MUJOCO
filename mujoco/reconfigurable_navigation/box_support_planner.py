"""State-aware rule plan for the accepted single-box support scene."""

import numpy as np

from .box_support_env import BOX_ID, PLATFORM_ID
from .oracle_planner import OraclePlanner, PlanResult
from .representations import SkillAction, SkillType


def operation_key(action):
    return action.skill, action.object_id if action.skill == SkillType.PUSH else action.support_id


class BoxSupportPlanner(OraclePlanner):
    def __init__(self, completed):
        super().__init__()
        self.completed = completed

    def plan(self, observation, push_target):
        objects = {obj.object_id: obj for obj in observation.objects}
        if not observation.state_valid or observation.illegal_collision:
            return PlanResult((), None, "invalid_box_support_state")
        if BOX_ID not in objects or PLATFORM_ID not in objects:
            return PlanResult((), None, "support_object_missing")
        box, platform = objects[BOX_ID], objects[PLATFORM_ID]
        if not box.movable or not box.supportable or not platform.supportable:
            return PlanResult((), None, "support_object_unavailable")
        if box.mass > observation.capability.max_pushable_mass:
            return PlanResult((), None, "box_exceeds_capability")
        if (SkillType.CLIMB, PLATFORM_ID) in self.completed:
            if np.linalg.norm(observation.robot_state[:2] - observation.goal[:2]) < 0.20:
                return PlanResult((SkillAction(SkillType.STOP),), None, "goal_reached")
            return PlanResult((), None, "goal_not_maintained")
        if (SkillType.PUSH, BOX_ID) in self.completed and np.linalg.norm(box.center[:2] - push_target[:2]) >= 0.12:
            self.completed.clear()
        box_top = box.center[2] + box.size[2] / 2.0
        platform_top = platform.center[2] + platform.size[2] / 2.0
        if max(box_top, platform_top - box_top) > observation.capability.max_step_height + 0.01:
            return PlanResult((), None, "support_rise_exceeds_capability")
        candidates = (
            SkillAction(SkillType.PUSH, push_target, object_id=BOX_ID),
            SkillAction(SkillType.NAV, box.climb_entry_pose, support_id=BOX_ID),
            SkillAction(SkillType.CLIMB, box.climb_landing_pose, support_id=BOX_ID),
            SkillAction(SkillType.NAV, platform.climb_entry_pose, support_id=PLATFORM_ID),
            SkillAction(SkillType.CLIMB, platform.climb_landing_pose, support_id=PLATFORM_ID),
        )
        remaining = tuple(action for action in candidates if operation_key(action) not in self.completed)
        return PlanResult(remaining, None, "box_support_progress" if remaining else "incomplete_support_plan")