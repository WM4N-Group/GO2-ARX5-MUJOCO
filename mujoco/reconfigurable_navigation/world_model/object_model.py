"""Masked object-centric dynamics with BEV context and conditional residuals."""

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from .encoding import BINARY_NAMES, FEATURE_NAMES, REGRESSION_NAMES
from .evaluation import conditional_baseline
from .model import _masked_mean, fit_normalization
from .spatial_encoding import MAX_OBJECTS, OBJECT_TOKEN_NAMES, PROPRIO_SIZE, SPATIAL_SCHEMA


def source_fingerprints():
    root = Path(__file__).parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in ("encoding.py", "spatial_encoding.py", "object_model.py")}


class ObjectWorldModel(nn.Module):
    def __init__(self, width=128, layers=4, heads=4, architecture="object_transformer"):
        super().__init__()
        if width % heads or width % 8 or layers < 1 or architecture not in ("object_transformer", "residual_mlp"):
            raise ValueError("Invalid object-model configuration")
        self.config = {"width": width, "layers": layers, "heads": heads, "architecture": architecture}
        for name, count, fill in (("feature_mean", len(FEATURE_NAMES), 0.0), ("feature_scale", len(FEATURE_NAMES), 1.0), ("target_mean", len(REGRESSION_NAMES), 0.0), ("target_scale", len(REGRESSION_NAMES), 1.0), ("object_mean", len(OBJECT_TOKEN_NAMES), 0.0), ("object_scale", len(OBJECT_TOKEN_NAMES), 1.0)):
            self.register_buffer(name, torch.full((count,), fill))
        self.register_buffer("prior_regression", torch.zeros(6, len(REGRESSION_NAMES)))
        self.register_buffer("prior_logits", torch.zeros(6, len(BINARY_NAMES)))
        self.register_buffer("proprio_mean", torch.zeros(PROPRIO_SIZE))
        self.register_buffer("proprio_scale", torch.ones(PROPRIO_SIZE))
        self.register_buffer("skill_indices", torch.tensor([FEATURE_NAMES.index(f"skill_{name}") for name in ("nav", "push", "climb")]))
        self.anchor_index = FEATURE_NAMES.index("anchor_platform")
        selections = {
            "robot": [index for index, name in enumerate(FEATURE_NAMES) if name.startswith(("robot_", "previous_", "completed_")) or name == "illegal_collision_before"],
            "goal": [index for index, name in enumerate(FEATURE_NAMES) if name.startswith(("goal_", "suffix_"))],
            "capability": [index for index, name in enumerate(FEATURE_NAMES) if name.startswith("capability_") or name == "box_friction"],
            "action": [index for index, name in enumerate(FEATURE_NAMES) if name.startswith(("target_", "skill_", "anchor_"))],
        }
        if architecture == "residual_mlp":
            self.tabular = nn.Sequential(nn.Linear(len(FEATURE_NAMES) + PROPRIO_SIZE, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        else:
            cnn = []
            channels = 8
            for output in (16, 32, 64, width):
                cnn.extend((nn.Conv2d(channels, output, 3, stride=2, padding=1), nn.GroupNorm(4 if output == 16 else 8, output), nn.SiLU()))
                channels = output
            self.scene_encoder = nn.Sequential(*cnn, nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.encoders = nn.ModuleDict()
            for name, selected in selections.items():
                self.register_buffer(f"{name}_indices", torch.tensor(selected))
                self.encoders[name] = nn.Sequential(nn.Linear(len(selected) + (PROPRIO_SIZE if name == "robot" else 0), width), nn.SiLU(), nn.Linear(width, width))
            self.object_projection = nn.Linear(len(OBJECT_TOKEN_NAMES), width)
            object_layer = nn.TransformerEncoderLayer(width, heads, width * 2, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
            self.object_encoder = nn.TransformerEncoder(object_layer, 1, enable_nested_tensor=False)
            dynamics_layer = nn.TransformerEncoderLayer(width, heads, width * 2, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
            self.dynamics = nn.TransformerEncoder(dynamics_layer, layers, enable_nested_tensor=False)
            self.type_embedding = nn.Embedding(6, width)
        self.robot_head = nn.Linear(width, 5)
        self.box_head = nn.Linear(width, 5)
        self.time_head = nn.Linear(width, 2)
        self.binary_head = nn.Linear(width, len(BINARY_NAMES))
        for head in (self.robot_head, self.box_head, self.time_head, self.binary_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def set_normalization(self, dataset):
        values = fit_normalization(dataset)
        for name, value in values.items():
            getattr(self, name).copy_(torch.from_numpy(value))
        train = dataset["split"] == "train"
        available = train & (dataset["proprio"][:, -1] > 0.5)
        if available.any():
            proprio = dataset["proprio"][available]
            self.proprio_mean.copy_(torch.from_numpy(proprio.mean(axis=0)))
            scale = proprio.std(axis=0)
            self.proprio_scale.copy_(torch.from_numpy(np.where(scale > 1e-3, scale, 1.0)))
        self.proprio_mean[-1] = 0.0
        self.proprio_scale[-1] = 1.0
        tokens = dataset["objects"][train][dataset["object_mask"][train]]
        scale = tokens.std(axis=0)
        self.object_mean.copy_(torch.from_numpy(tokens.mean(axis=0)))
        self.object_scale.copy_(torch.from_numpy(np.where(scale > 1e-3, scale, 1.0)))
        regression, probabilities = conditional_baseline(dataset)
        global_labels, global_mask = dataset["binary"][train], dataset["binary_mask"][train]
        global_probability = ((global_labels * global_mask).sum(axis=0) + 0.5) / (global_mask.sum(axis=0) + 1.0)
        for group in range(6):
            members = np.flatnonzero(dataset["group"] == group)
            mean = regression[members[0]] if len(members) else values["target_mean"]
            probability = probabilities[members[0]] if len(members) else global_probability
            self.prior_regression[group].copy_(torch.as_tensor(mean))
            self.prior_logits[group].copy_(torch.as_tensor(np.log(probability / (1.0 - probability))))

    def forward(self, features, bev, objects, object_mask, object_ids, proprio=None):
        normalized = (features - self.feature_mean) / self.feature_scale
        if proprio is None:
            proprio = features.new_zeros((len(features), PROPRIO_SIZE))
        normalized_proprio = ((proprio - self.proprio_mean) / self.proprio_scale).masked_fill(proprio[:, -1:] < 0.5, 0.0)
        group = features[:, self.skill_indices].argmax(dim=1) * 2 + features[:, self.anchor_index].long()
        if self.config["architecture"] == "residual_mlp":
            robot_latent = box_latent = action_latent = self.tabular(torch.cat((normalized, normalized_proprio), dim=1))
        else:
            box_locations = (object_ids == 10) & object_mask
            if not torch.all(box_locations.sum(dim=1) == 1):
                raise ValueError("Exactly one visible box role (ID 10) is required")
            objects = ((objects - self.object_mean) / self.object_scale).masked_fill(~object_mask.unsqueeze(-1), 0.0)
            object_latent = self.object_encoder(self.object_projection(objects), src_key_padding_mask=~object_mask)
            special = [self.scene_encoder(bev.to(features.dtype))]
            for name in ("robot", "goal", "capability", "action"):
                values = normalized[:, getattr(self, f"{name}_indices")]
                if name == "robot":
                    values = torch.cat((values, normalized_proprio), dim=1)
                special.append(self.encoders[name](values))
            tokens = torch.cat((torch.stack(special, dim=1), object_latent), dim=1)
            kinds = torch.cat((torch.arange(5, device=features.device), torch.full((objects.shape[1],), 5, device=features.device)))
            padding = torch.cat((torch.zeros((len(features), 5), dtype=torch.bool, device=features.device), ~object_mask), dim=1)
            latent = self.dynamics(tokens + self.type_embedding(kinds).unsqueeze(0), src_key_padding_mask=padding)
            robot_latent, action_latent = latent[:, 1], latent[:, 4]
            box_indices = box_locations.long().argmax(dim=1)
            box_latent = latent[torch.arange(len(features), device=features.device), box_indices + 5]
        residual = torch.cat((self.robot_head(robot_latent), self.box_head(box_latent), self.time_head(action_latent)), dim=1)
        return self.prior_regression[group] + residual * self.target_scale, self.prior_logits[group] + self.binary_head(action_latent)

    def loss(self, inputs, regression, regression_mask, binary, binary_mask):
        prediction, logits = self(*inputs)
        regression_loss = functional.smooth_l1_loss((prediction - self.target_mean) / self.target_scale, (regression - self.target_mean) / self.target_scale, reduction="none")
        binary_loss = functional.binary_cross_entropy_with_logits(logits, binary, reduction="none")
        return _masked_mean(regression_loss, regression_mask) + _masked_mean(binary_loss, binary_mask)

    def save(self, path, metadata):
        torch.save({"schema": SPATIAL_SCHEMA, "config": self.config, "source_sha256": source_fingerprints(), "object_token_names": OBJECT_TOKEN_NAMES, "max_objects": MAX_OBJECTS, "state_dict": self.state_dict(), "metadata": metadata}, path)

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["schema"] != SPATIAL_SCHEMA or checkpoint["source_sha256"] != source_fingerprints():
            raise ValueError("Incompatible object-world-model encoding or implementation")
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model, checkpoint["metadata"]