"""Small masked multi-head MLP, preceding the planned object Transformer."""

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from .encoding import BINARY_NAMES, FEATURE_NAMES, REGRESSION_NAMES, SCHEMA, encode_inputs


def fit_normalization(dataset):
    train = dataset["split"] == "train"
    if not train.any():
        raise ValueError("Training split is empty")
    features = dataset["features"][train]
    feature_scale = features.std(axis=0)
    targets, mask = dataset["regression"][train], dataset["regression_mask"][train]
    counts = mask.sum(axis=0)
    target_mean = (targets * mask).sum(axis=0) / np.maximum(counts, 1)
    target_scale = np.sqrt((((targets - target_mean) ** 2) * mask).sum(axis=0) / np.maximum(counts, 1))
    return {
        "feature_mean": features.mean(axis=0).astype(np.float32),
        "feature_scale": np.where(feature_scale > 1e-3, feature_scale, 1.0).astype(np.float32),
        "target_mean": target_mean.astype(np.float32),
        "target_scale": np.where(counts > 0, np.maximum(target_scale, 0.05), 1.0).astype(np.float32),
    }


def _masked_mean(values, mask):
    counts = mask.sum(dim=0)
    per_head = (values * mask).sum(dim=0) / counts.clamp(min=1)
    return per_head[counts > 0].mean() if (counts > 0).any() else values.sum() * 0.0


class SkillWorldModel(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        for name, count, fill in (("feature_mean", len(FEATURE_NAMES), 0.0), ("feature_scale", len(FEATURE_NAMES), 1.0), ("target_mean", len(REGRESSION_NAMES), 0.0), ("target_scale", len(REGRESSION_NAMES), 1.0)):
            self.register_buffer(name, torch.full((count,), fill))
        self.network = nn.Sequential(nn.Linear(len(FEATURE_NAMES), hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, len(REGRESSION_NAMES) + len(BINARY_NAMES)))

    def set_normalization(self, values):
        for name, value in values.items():
            getattr(self, name).copy_(torch.from_numpy(value))

    def forward(self, features):
        output = self.network((features - self.feature_mean) / self.feature_scale)
        return output[:, :len(REGRESSION_NAMES)] * self.target_scale + self.target_mean, output[:, len(REGRESSION_NAMES):]

    def loss(self, features, regression, regression_mask, binary, binary_mask):
        predicted, logits = self(features)
        regression_loss = functional.smooth_l1_loss((predicted - self.target_mean) / self.target_scale, (regression - self.target_mean) / self.target_scale, reduction="none")
        binary_loss = functional.binary_cross_entropy_with_logits(logits, binary, reduction="none")
        return _masked_mean(regression_loss, regression_mask) + _masked_mean(binary_loss, binary_mask)

    @torch.inference_mode()
    def predict(self, observation, action, **context):
        features = encode_inputs(observation, action, **context)
        regression, logits = self(torch.from_numpy(features).unsqueeze(0).to(self.feature_mean.device))
        values = regression[0].cpu().numpy()
        probabilities = torch.sigmoid(logits[0]).cpu().numpy()
        return {
            "robot_position_delta": values[:3].tolist(),
            "robot_yaw_delta": float(np.arctan2(values[3], values[4])),
            "box_position_delta": values[5:8].tolist(),
            "box_yaw_delta": float(np.arctan2(values[8], values[9])),
            "skill_seconds": float(np.expm1(np.clip(values[10], 0, 80))),
            "total_attempt_seconds": float(np.expm1(np.clip(values[11], 0, 80))),
            **{f"{name}_probability": float(value) for name, value in zip(BINARY_NAMES, probabilities)},
        }

    def save(self, path, metadata):
        torch.save({"schema": SCHEMA, "feature_names": FEATURE_NAMES, "regression_names": REGRESSION_NAMES, "binary_names": BINARY_NAMES, "encoding_sha256": hashlib.sha256(Path(__file__).with_name("encoding.py").read_bytes()).hexdigest(), "hidden_size": self.hidden_size, "state_dict": self.state_dict(), "metadata": metadata}, path)

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["schema"] != SCHEMA or tuple(checkpoint["feature_names"]) != FEATURE_NAMES or checkpoint["encoding_sha256"] != hashlib.sha256(Path(__file__).with_name("encoding.py").read_bytes()).hexdigest():
            raise ValueError("Incompatible world-model input encoding")
        model = cls(checkpoint["hidden_size"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model, checkpoint["metadata"]