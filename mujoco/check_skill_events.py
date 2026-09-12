"""Check process event and termination contracts using actual skill logic."""

from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from reconfigurable_navigation.representations import (
    Capability, ObjectState, ObjectType, OracleObservation, SkillAction, SkillType,
)
from reconfigurable_navigation.skills import SkillStatus
from reconfigurable_navigation.skills.climb import ClimbConfig, ClimbSkill
from reconfigurable_navigation.skills.navigate import NavigateConfig, NavigateSkill
from reconfigurable_navigation.skills.push import PushConfig, PushSkill


class SkillEventChecks(unittest.TestCase):
    def setUp(self) -> None:
        self.box = ObjectState(
            object_id=10, object_type=ObjectType.MOVABLE_BOX,
            center=np.array([0.7, 0.0, 0.25]), size=np.ones(3), movable=True, mass=5.0,
        )
        self.observation = OracleObservation(
            robot_state=np.array([0.0, 0.0, 0.3] + [0.0] * 9),
            goal=np.array([2.0, 0.0, 0.0, 1.0]),
            objects=(self.box,), capability=Capability(),
        )
        self.actions = (
            SkillAction(SkillType.NAV, np.array([2.0, 0.0, 0.0])),
            SkillAction(SkillType.PUSH, np.array([1.7, 0.0, 0.0]), object_id=10),
            SkillAction(SkillType.CLIMB, np.array([2.0, 0.0, 0.5, 0.0])),
        )

    def test_invalid_state_and_reset_reason_for_every_skill(self) -> None:
        for skill, action in zip((NavigateSkill(), PushSkill(), ClimbSkill()), self.actions):
            with self.subTest(skill=action.skill):
                skill.reset(action)
                skill.step(replace(self.observation, state_valid=False))
                self.assertEqual(skill.status, SkillStatus.FAILED)
                self.assertEqual(skill.failure_reason, "invalid_robot_state")
                skill.reset(action)
                self.assertIsNone(skill.failure_reason)

    def test_real_skill_timeouts(self) -> None:
        skills = (
            NavigateSkill(NavigateConfig(timeout_steps=1)),
            PushSkill(PushConfig(timeout_steps=1)),
            ClimbSkill(ClimbConfig(timeout_steps=1)),
        )
        for skill, action in zip(skills, self.actions):
            with self.subTest(skill=action.skill):
                skill.reset(action)
                for _ in range(2):
                    skill.step(self.observation)
                    if skill.status != SkillStatus.RUNNING:
                        break
                self.assertEqual(skill.failure_reason, "timeout")
                self.assertEqual(skill.status, SkillStatus.FAILED)

    def test_collision_reason_for_push_and_climb(self) -> None:
        for skill, action in zip((PushSkill(), ClimbSkill()), self.actions[1:]):
            skill.reset(action)
            skill.step(replace(self.observation, illegal_collision=True))
            self.assertEqual(skill.failure_reason, "illegal_collision")

    def test_push_object_and_body_failures(self) -> None:
        cases = (
            (replace(self.observation, objects=()), "object_missing"),
            (replace(self.observation, objects=(replace(self.box, movable=False),)), "object_not_movable"),
            (replace(self.observation, objects=(replace(self.box, mass=100.0),)), "object_too_heavy"),
            (replace(self.observation, body_contact_object_ids=(10,)), "body_contact"),
        )
        for observation, reason in cases:
            with self.subTest(reason=reason):
                skill = PushSkill()
                skill.reset(self.actions[1])
                skill.step(observation)
                self.assertEqual(skill.failure_reason, reason)

    def test_contact_timeout_is_distinct_from_no_progress(self) -> None:
        for contact, expected in ((False, "contact_timeout"), (True, "no_progress")):
            with self.subTest(contact=contact):
                skill = PushSkill(PushConfig(
                    align_steps=1, contact_timeout_steps=1, no_progress_timeout_steps=1,
                ))
                skill.reset(self.actions[1])
                observation = replace(
                    self.observation, end_effector_contact_object_ids=(10,) if contact else (),
                )
                for _ in range(5):
                    skill.step(observation)
                    if skill.status == SkillStatus.FAILED:
                        break
                self.assertEqual(skill.failure_reason, expected)

    def test_push_verification_detects_target_loss(self) -> None:
        skill = PushSkill(PushConfig(align_steps=1, verify_steps=1))
        skill.reset(self.actions[1])
        contact = replace(self.observation, end_effector_contact_object_ids=(10,))
        skill.step(contact)
        skill.step(contact)
        reached = replace(contact, objects=(replace(self.box, center=np.array([1.7, 0.0, 0.25])),))
        skill.step(reached)
        skill.step(contact)
        self.assertEqual(skill.failure_reason, "target_not_maintained")


if __name__ == "__main__":
    unittest.main()