"""Equal-episode accounting for parallel skill evaluations."""

import torch


class EpisodeQuota:
    def __init__(self, num_envs, episodes_per_env, device):
        if num_envs < 1 or episodes_per_env < 1:
            raise ValueError("Evaluation quotas must be positive")
        self.quota = episodes_per_env
        self.completed = torch.zeros(num_envs, dtype=torch.int64, device=device)
        self.successful = torch.zeros_like(self.completed)

    @property
    def active(self):
        return self.completed < self.quota

    def record(self, dones, successes):
        counted = self.active & dones.bool()
        self.completed += counted.long()
        self.successful += (counted & successes.bool()).long()
        return counted

    def terminal_snapshot(self, env_ids, **fields):
        selected = env_ids[self.active[env_ids]]
        values = {name: tensor[selected].tolist() for name, tensor in fields.items()}
        episodes = (self.completed[selected] + 1).tolist()
        return [
            {"env_id": env_id, "episode": episodes[index], **{name: value[index] for name, value in values.items()}}
            for index, env_id in enumerate(selected.tolist())
        ]