"""Check skill-boundary recording contracts with deterministic test doubles."""

from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from reconfigurable_navigation.representations import (
    Capability,
    ObjectState,
    ObjectType,
    OracleObservation,
    SkillAction,
    SkillType,
)
from reconfigurable_navigation.runtime import ReconfigurableExecutor
from reconfigurable_navigation.skills import Skill, SkillCommand, SkillStatus


class TestEnvironment:
    def __init__(self) -> None:
        self.model = object()
        self.data = SimpleNamespace(time=0.0)
        self.observation = OracleObservation(
            robot_state=np.zeros(12, dtype=np.float32),
            goal=np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32),
            objects=(
                ObjectState(
                    object_id=10,
                    object_type=ObjectType.MOVABLE_BOX,
                    center=np.zeros(3, dtype=np.float32),
                    size=np.ones(3, dtype=np.float32),
                    movable=True,
                ),
            ),
            capability=Capability(),
        )

    def observe(self) -> OracleObservation:
        return self.observation


class TestRuntime:
    control_dt = 0.02

    def __init__(self, env: TestEnvironment) -> None:
        self.env = env
        self.model = env.model
        self.data = env.data

    def hold_default(self) -> None:
        self.data.time += self.control_dt

    def step(self, _velocity: np.ndarray, _end_effector_pose: np.ndarray | None = None) -> None:
        self.data.time += self.control_dt
        self.env.observation.robot_state[0] += 1.0
        self.env.observation.objects[0].center[0] += 1.0

    def reset(self) -> None:
        pass

    def activate(self) -> None:
        pass


class TransitionChecks(unittest.TestCase):
    def execute(
        self,
        skill_type: SkillType = SkillType.NAV,
        interrupt: bool = False,
        status: SkillStatus = SkillStatus.SUCCEEDED,
        record: bool = True,
        running_steps: int = 0,
    ):
        env = TestEnvironment()
        runtime = TestRuntime(env)
        action = SkillAction(skill_type, np.ones(4 if skill_type == SkillType.CLIMB else 3))
        stop = SkillAction(SkillType.STOP)
        replanner = Mock()
        replanner.decide.side_effect = [
            SimpleNamespace(
                plan=SimpleNamespace(actions=(action, stop)),
                reason="test_action",
                terminal=False,
                succeeded=False,
                action=action,
            ),
            SimpleNamespace(
                plan=SimpleNamespace(actions=(stop,)),
                reason="goal_reached",
                terminal=True,
                succeeded=True,
                action=stop,
            ),
        ]
        skill = Mock(spec=Skill)
        skill.can_execute.return_value = True
        skill.config = SimpleNamespace(timeout_steps=max(1, running_steps + 1))
        skill.status = SkillStatus.RUNNING
        skill.failure_reason = None
        statuses = iter([SkillStatus.RUNNING] * running_steps + [status])

        def advance_skill(_observation):
            skill.status = next(statuses)
            return SkillCommand(np.zeros(3), np.zeros(7))

        skill.step.side_effect = advance_skill
        executor = ReconfigurableExecutor(
            env,
            runtime,
            climb_runtime=TestRuntime(env),
            replanner=replanner,
        )
        transitions = []
        with patch.object(executor, "_create_skill", return_value=skill):
            result = executor.run(
                on_step=lambda _action, active_skill, _observation: not interrupt or active_skill is None,
                on_transition=transitions.append if record else None,
            )
        return env, action, result, transitions

    def test_successful_task_can_contain_a_failed_skill(self) -> None:
        _env, _action, result, transitions = self.execute(status=SkillStatus.FAILED)
        self.assertTrue(result.succeeded)
        self.assertEqual(result.skill_failures, 1)
        payload = json.loads(transitions[0].to_json())
        self.assertEqual(payload["outcome"], "failed")
        self.assertIs(payload["skill_success"], False)

    def test_recording_does_not_change_execution_result(self) -> None:
        recorded_env, _action, recorded_result, transitions = self.execute()
        plain_env, _action, plain_result, no_transitions = self.execute(record=False)
        self.assertEqual(no_transitions, [])
        self.assertEqual(len(transitions), 1)
        self.assertEqual(recorded_result.succeeded, plain_result.succeeded)
        self.assertEqual(recorded_result.reason, plain_result.reason)
        self.assertEqual(recorded_result.replans, plain_result.replans)
        self.assertEqual(recorded_result.records[0].steps, plain_result.records[0].steps)
        self.assertAlmostEqual(recorded_env.data.time, plain_env.data.time)
        np.testing.assert_array_equal(
            recorded_env.observation.robot_state,
            plain_env.observation.robot_state,
        )

    def test_observations_and_actions_are_detached(self) -> None:
        env, action, result, transitions = self.execute()
        self.assertTrue(result.succeeded)
        self.assertEqual(len(transitions), len(result.records))
        transition = transitions[0]
        env.observation.robot_state[0] = 99.0
        env.observation.objects[0].center[0] = 99.0
        action.target_pose[0] = 99.0
        self.assertEqual(transition.observation_before.robot_state[0], 0.0)
        self.assertEqual(transition.observation_after.robot_state[0], 1.0)
        self.assertEqual(transition.observation_after.objects[0].center[0], 1.0)
        self.assertEqual(transition.action.target_pose[0], 1.0)
        self.assertAlmostEqual(transition.elapsed_sim_time, 0.02)

    def test_interruption_is_not_a_physical_failure_label(self) -> None:
        _env, _action, result, transitions = self.execute(interrupt=True)
        self.assertEqual(result.reason, "execution_interrupted")
        self.assertEqual(result.skill_failures, 0)
        payload = json.loads(transitions[0].to_json())
        self.assertEqual(payload["outcome"], "interrupted")
        self.assertIsNone(payload["skill_success"])
        self.assertEqual(payload["termination_reason"], "execution_interrupted")
        self.assertIsNone(payload["failure_reason"])

    def test_climb_preparation_is_included_in_elapsed_time(self) -> None:
        _env, _action, result, transitions = self.execute(SkillType.CLIMB)
        self.assertTrue(result.succeeded)
        self.assertEqual(transitions[0].skill_steps, 1)
        self.assertAlmostEqual(transitions[0].elapsed_sim_time, 0.52)
        self.assertEqual(transitions[0].event_sample_count, 27)

    def test_transient_contacts_and_collision_survive_final_recovery(self) -> None:
        original_step = TestRuntime.step
        updates = iter([
            dict(
                contact_object_ids=(10,),
                end_effector_contact_object_ids=(10,),
                illegal_collision=True,
            ),
            dict(
                contact_object_ids=(),
                end_effector_contact_object_ids=(),
                illegal_collision=False,
            ),
            {},
        ])

        def advance(runtime, velocity, end_effector_pose=None):
            original_step(runtime, velocity, end_effector_pose)
            runtime.env.observation = replace(
                runtime.env.observation, **next(updates, {})
            )

        with patch.object(TestRuntime, "step", new=advance):
            _env, _action, result, transitions = self.execute(running_steps=2)
        payload = json.loads(transitions[0].to_json())
        self.assertTrue(result.succeeded)
        self.assertFalse(payload["observation_after"]["illegal_collision"])
        collisions = [event for event in payload["events"] if event["kind"] == "illegal_collision"]
        self.assertEqual([event["active"] for event in collisions], [True, False])
        self.assertEqual([event["control_step"] for event in collisions], [1, 2])
        self.assertEqual(payload["event_sample_count"], 4)
        self.assertTrue(payload["process_labels"]["illegal_collision"])
        self.assertTrue(payload["process_labels"]["end_effector_contact"])

    def test_json_handles_numpy_and_enum_values(self) -> None:
        _env, _action, _result, transitions = self.execute()
        payload = json.loads(transitions[0].to_json({"seed": np.int64(0)}))
        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(payload["action"]["skill"], int(SkillType.NAV))
        self.assertEqual(payload["metadata"]["seed"], 0)
        self.assertEqual(len(payload["observation_before"]["robot_state"]), 12)
        self.assertTrue(payload["skill_success"])

    def test_non_finite_values_are_not_written_as_training_json(self) -> None:
        _env, _action, _result, transitions = self.execute()
        transitions[0].observation_after.robot_state[0] = np.nan
        with self.assertRaises(ValueError):
            transitions[0].to_json()


if __name__ == "__main__":
    unittest.main()