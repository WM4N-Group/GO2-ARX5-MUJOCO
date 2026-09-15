"""Check alternate skill execution without changing planner and recording semantics."""

from types import SimpleNamespace
import unittest

import numpy as np

from reconfigurable_navigation.representations import Capability, OracleObservation, SkillAction, SkillType
from reconfigurable_navigation.runtime.executor import ReconfigurableExecutor
from reconfigurable_navigation.runtime.safety import SafetyConfig, SafetyMonitor
from reconfigurable_navigation.skills import NavigateSkill, SkillStatus


class Environment:
    def __init__(self):
        self.data = SimpleNamespace(time=0.0)
        self.push_target = np.array([1.0, 0.0, 0.0])
        self.position = 0.0

    def observe(self):
        robot = np.zeros(12, dtype=np.float32)
        robot[[0, 2]] = [self.position, 0.33]
        return OracleObservation(robot, self.push_target.copy(), (), Capability())


class Planner:
    def decide(self, observation):
        terminal = observation.robot_state[0] >= 1.0
        action = SkillAction(SkillType.STOP if terminal else SkillType.NAV, np.array([1.0, 0.0, 0.0]))
        return SimpleNamespace(plan=SimpleNamespace(actions=(action,)), action=action, terminal=terminal, succeeded=terminal, reason="goal_reached" if terminal else "navigate")


class Backend:
    def __init__(self, env, fail=False, reject=False):
        self.env = env
        self.fail = fail
        self.reject = reject

    def initial_stage(self, action):
        return "execution"

    def create_skill(self, action):
        return None if self.reject else NavigateSkill()

    def execute_skill(self, action, skill, after_step):
        self.env.data.time += 0.02
        self.env.position = 0.5 if self.fail else 1.0
        if not after_step("execution", "backend_motion"):
            skill.status = SkillStatus.FAILED
            return skill.status, 1, True
        skill.status = SkillStatus.FAILED if self.fail else SkillStatus.SUCCEEDED
        skill.failure_reason = "physical_failure" if self.fail else None
        return skill.status, 1, False


class BackendChecks(unittest.TestCase):
    def make_executor(self, *, fail=False, reject=False):
        env = Environment()

        def forbidden(*args):
            raise AssertionError("Legacy runtime must not advance custom execution")

        runtime = SimpleNamespace(control_dt=0.02, hold_default=forbidden, step=forbidden)
        return ReconfigurableExecutor(env, runtime, replanner=Planner(), safety=SafetyMonitor(SafetyConfig(max_skill_failures=1)), skill_backend=Backend(env, fail, reject), stabilization_duration=0.0, terminal_hold=False)

    def test_backend_preserves_skill_start_transition_and_replanning(self):
        executor = self.make_executor()
        starts, transitions = [], []
        result = executor.run(on_skill_start=lambda action, previous: starts.append((action, previous, executor.env.data.time)), on_transition=transitions.append)
        self.assertTrue(result.succeeded)
        self.assertEqual(result.replans, 2)
        self.assertEqual(len(starts), 1)
        self.assertEqual(starts[0][2], 0.0)
        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0].observation_before.robot_state[0], 0.0)
        self.assertEqual(transitions[0].observation_after.robot_state[0], 1.0)
        self.assertEqual(transitions[0].event_sample_count, 2)
        self.assertAlmostEqual(transitions[0].ended_at - transitions[0].started_at, 0.02)

    def test_callback_interruption_is_not_a_skill_failure(self):
        transitions = []
        result = self.make_executor().run(on_step=lambda *args: False, on_transition=transitions.append)
        self.assertEqual(result.reason, "execution_interrupted")
        self.assertEqual(result.skill_failures, 0)
        self.assertTrue(transitions[0].interrupted)
        self.assertEqual(transitions[0].termination_reason, "execution_interrupted")

    def test_physical_failure_is_recorded_before_budget_stops_execution(self):
        transitions = []
        result = self.make_executor(fail=True).run(on_transition=transitions.append)
        self.assertEqual(result.reason, "skill_failure_budget_exhausted")
        self.assertEqual(result.skill_failures, 1)
        self.assertEqual(transitions[0].termination_reason, "physical_failure")
        self.assertFalse(transitions[0].interrupted)

    def test_rejected_action_has_no_physical_step_or_transition(self):
        executor = self.make_executor(reject=True)
        starts, transitions = [], []
        result = executor.run(on_skill_start=lambda *args: starts.append(args), on_transition=transitions.append)
        self.assertEqual(result.reason, "skill_not_executable")
        self.assertEqual(executor.env.data.time, 0.0)
        self.assertEqual(starts, [])
        self.assertEqual(transitions, [])


if __name__ == "__main__":
    unittest.main()