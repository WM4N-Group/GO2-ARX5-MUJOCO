"""Validate equal weighting of successes, failures and incomplete trials."""

import unittest

import torch

from box_evaluation_metrics import EpisodeQuota


class EvaluationChecks(unittest.TestCase):
    def test_fast_failures_do_not_outweigh_slow_success(self):
        quota = EpisodeQuota(2, 1, "cpu")
        for _ in range(5):
            quota.record(torch.tensor([True, False]), torch.tensor([False, False]))
        quota.record(torch.tensor([True, True]), torch.tensor([False, True]))
        torch.testing.assert_close(quota.completed, torch.tensor([1, 1]))
        torch.testing.assert_close(quota.successful, torch.tensor([0, 1]))
        self.assertFalse(quota.active.any())

    def test_incomplete_trials_remain_distinct(self):
        quota = EpisodeQuota(2, 2, "cpu")
        quota.record(torch.tensor([True, False]), torch.tensor([True, True]))
        torch.testing.assert_close(quota.completed, torch.tensor([1, 0]))
        torch.testing.assert_close(quota.successful, torch.tensor([1, 0]))
        self.assertTrue(quota.active.all())

    def test_terminal_snapshot_survives_reset_and_ignores_finished_envs(self):
        quota = EpisodeQuota(2, 1, "cpu")
        quota.record(torch.tensor([True, False]), torch.tensor([True, False]))
        position = torch.tensor([[0.0, 0.0], [1.6, 0.1]])
        snapshot = quota.terminal_snapshot(torch.tensor([0, 1]), box_position=position)
        position.zero_()
        quota.record(torch.tensor([True, True]), torch.tensor([True, True]))
        self.assertEqual(len(snapshot), 1)
        self.assertEqual(snapshot[0]["env_id"], 1)
        self.assertEqual(snapshot[0]["episode"], 1)
        torch.testing.assert_close(torch.tensor(snapshot[0]["box_position"]), torch.tensor([1.6, 0.1]))
        self.assertEqual(quota.terminal_snapshot(torch.tensor([0, 1]), box_position=position), [])


if __name__ == "__main__":
    unittest.main()