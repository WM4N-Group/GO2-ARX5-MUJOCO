"""Validate robot-specific provenance before exporting a trained skill."""

import hashlib
from pathlib import Path

import yaml


def piper_robot_provenance(root, checkpoint, evaluations):
    config = yaml.load((checkpoint.parent / "params/env.yaml").read_text(), Loader=yaml.BaseLoader)
    usd = Path(config["scene"]["robot"]["spawn"]["usd_path"])
    if usd.parent.name != "go2_piper":
        raise ValueError("Training configuration does not use PIPER")
    reference = None
    for report in evaluations:
        if report.get("observation_noise_diagnostic"):
            raise ValueError("Noise-injected diagnostic cannot establish deterministic acceptance")
        physics = report["actual_physics"]
        assets = physics.get("robot_asset_sha256")
        if not assets:
            raise ValueError("Evaluation lacks robot asset provenance")
        for name, expected in assets.items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts or "go2_piper" not in relative.parts:
                raise ValueError("Invalid PIPER asset reference")
            path = root / relative
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected:
                raise ValueError(f"PIPER asset differs from evaluation: {name}")
        record = {"robot": "go2_piper", "assets_sha256": assets,
                  "joint_names": physics["robot_joint_names"], "effort_limits": physics["robot_effort_limits"],
                  "joint_defaults": physics["robot_joint_defaults"],
                  "controller_contract": physics.get("controller_contract"),
                  "preparation_seconds": report.get("preparation_seconds", 0.0)}
        if len(record["joint_names"]) != 18 or len(record["effort_limits"]) != 18:
            raise ValueError("Incomplete PIPER actuator provenance")
        if reference is not None and record != reference:
            raise ValueError("Evaluation robot profiles differ")
        reference = record
    return reference