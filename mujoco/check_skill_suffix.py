"""Check continuation budgets, cost accounting and conservative labels."""

from copy import deepcopy
from types import SimpleNamespace
import unittest

import check_execution_backend as execution_checks
from reconfigurable_navigation.data.snapshot import SkillReplay, _rng_state
from reconfigurable_navigation.data.suffix import candidate_continuation, evaluate_suffix
from reconfigurable_navigation.skills import SkillStatus


class Snapshot:
    def __init__(self, **kwargs):
        self.executor = execution_checks.BackendChecks().make_executor(**kwargs)
        self._random_state = _rng_state()

    def fork(self):
        return deepcopy(self.executor)


class PreparedBackend(execution_checks.Backend):
    def execute_skill(self, action, skill, after_step):
        for index in range(3):
            self.env.data.time += 0.02
            if not after_step("preparation" if index < 2 else "execution"):
                return SkillStatus.FAILED, 0, True
        self.env.position = 1.0
        skill.status = SkillStatus.SUCCEEDED
        return skill.status, 1, False


class SuffixChecks(unittest.TestCase):
    def test_success_proves_reachability_without_mutating_source(self):
        snapshot = Snapshot()
        result = evaluate_suffix(snapshot)
        self.assertTrue(result["oracle_task_success"])
        self.assertTrue(result["reachability"])
        self.assertEqual(result["suffix_control_steps"], 1)
        self.assertEqual(snapshot.executor.env.data.time, 0.0)
        self.assertEqual(snapshot.executor.env.position, 0.0)

    def test_failure_and_rejection_do_not_prove_unreachability(self):
        for options in ({"fail": True}, {"reject": True}):
            result = evaluate_suffix(Snapshot(**options))
            self.assertIs(result["oracle_task_success"], False)
            self.assertIsNone(result["reachability"])

    def test_budget_exhaustion_stays_unknown_and_respects_control_limit(self):
        for budget in (0, 1):
            result = evaluate_suffix(Snapshot(), max_control_steps=budget)
            self.assertIsNone(result["oracle_task_success"])
            self.assertIsNone(result["reachability"])
            self.assertEqual(result["suffix_control_steps"], budget)
        self.assertIsNone(evaluate_suffix(Snapshot(), max_skills=0)["oracle_task_success"])

    def test_already_completed_task_needs_no_remaining_budget(self):
        snapshot = Snapshot()
        snapshot.executor.env.position = 1.0
        self.assertTrue(evaluate_suffix(snapshot, max_control_steps=0, max_skills=0)["oracle_task_success"])

    def test_cost_includes_preparation_and_candidate_time(self):
        snapshot = Snapshot()
        snapshot.executor.skill_backend = PreparedBackend(snapshot.executor.env)
        transition = SimpleNamespace(started_at=1.0, ended_at=1.1, interrupted=False, status=SkillStatus.SUCCEEDED)
        result = candidate_continuation(SkillReplay(transition, snapshot, "succeeded"))
        self.assertAlmostEqual(result["total_sim_time"], 0.16)
        self.assertEqual(result["suffix_control_steps"], 3)
        self.assertTrue(result["label_validity"]["total_cost"])

    def test_rejected_and_truncated_candidates_have_no_negative_task_label(self):
        rejected = candidate_continuation(SkillReplay(None, None, "unsupported_skill"))
        self.assertIsNone(rejected["total_sim_time"])
        self.assertFalse(rejected["label_validity"]["oracle_task_success"])
        transition = SimpleNamespace(started_at=1.0, ended_at=1.1, interrupted=True, status=SkillStatus.FAILED)
        result = candidate_continuation(SkillReplay(transition, None, "rollout_budget_exhausted"))
        self.assertIsNone(result["oracle_task_success"])
        self.assertTrue(result["cost_is_lower_bound"])
        self.assertFalse(result["label_validity"]["total_cost"])


if __name__ == "__main__":
    unittest.main()