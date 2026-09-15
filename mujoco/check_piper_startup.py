"""Check the existing GO2-PIPER assets and policy interfaces without a viewer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parent
DEPLOY = ROOT / "deploy/deploy_mujoco/go2_piper"
JOINT_NAMES = tuple(
    f"{leg}_{joint}_joint"
    for leg in ("FL", "FR", "RL", "RR")
    for joint in ("hip", "thigh", "calf")
) + tuple(f"joint{index}" for index in range(1, 7))


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def inspect_configuration(config_path: Path) -> dict:
    with config_path.open() as source:
        config = yaml.safe_load(source)
    policy_path = Path(config["policy_path"].replace("{CURRENT_ROOT_DIR}", str(ROOT)))
    model_path = Path(config["xml_path"].replace("{CURRENT_ROOT_DIR}", str(ROOT)))
    model = mujoco.MjModel.from_xml_path(str(model_path))
    if model.nu != 18 or config["num_actions"] != 18 or config["num_hist"] != 3:
        raise ValueError("Expected 18 actuators/actions and three history frames")
    joint_ids = np.array([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in JOINT_NAMES
    ])
    if np.any(joint_ids < 0):
        raise ValueError("Missing PIPER robot joints")
    if not np.array_equal(model.actuator_trnid[:, 0], joint_ids):
        raise ValueError("Actuator order differs from the configured robot joint order")
    for name in ("default_angles", "kps", "kds", "action_scale"):
        values = np.asarray(config[name], dtype=np.float64)
        if values.shape != (18,) or not np.isfinite(values).all():
            raise ValueError(f"{name} must contain 18 finite values")
    default_angles = np.asarray(config["default_angles"])
    joint_ranges = model.jnt_range[joint_ids]
    if np.any(default_angles < joint_ranges[:, 0]) or np.any(default_angles > joint_ranges[:, 1]):
        raise ValueError("Nominal pose violates joint limits")

    policy = torch.jit.load(str(policy_path), map_location="cpu").eval()
    generator = torch.Generator().manual_seed(0)
    probes = torch.cat((torch.zeros(1, 210), torch.randn(3, 210, generator=generator)))
    with torch.inference_mode():
        actions = torch.cat([policy(probe.unsqueeze(0)) for probe in probes])
    if actions.shape != (4, 18) or not torch.isfinite(actions).all():
        raise ValueError("Policy must produce 18 finite actions from 210 observations")

    initial_angles = model.qpos0[model.jnt_qposadr[joint_ids]]
    initial_limit_mask = (initial_angles < joint_ranges[:, 0]) | (initial_angles > joint_ranges[:, 1])
    return {
        "passed": True,
        "configuration": str(config_path),
        "configuration_sha256": sha256(config_path),
        "policy": str(policy_path),
        "policy_sha256": sha256(policy_path),
        "scene": str(model_path),
        "scene_sha256": sha256(model_path),
        "robot_xml_sha256": sha256(ROOT / "robots/go2_piper/go2piper.xml"),
        "nq": model.nq,
        "nv": model.nv,
        "actuators": model.nu,
        "robot_mass_kg": float(model.body_mass.sum()),
        "joint_names": JOINT_NAMES,
        "observation_size": 210,
        "action_size": 18,
        "probe_max_abs_action": float(actions.abs().max()),
        "control_dt": float(config["simulation_dt"] * config["control_decimation"]),
        "initial_joint_limit_violations": [
            name for name, invalid in zip(JOINT_NAMES, initial_limit_mask) if invalid
        ],
        "initial_max_nominal_error_rad": float(np.max(np.abs(initial_angles - default_angles))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.output_json is not None and args.output_json.exists():
        parser.error(f"Refusing to overwrite {args.output_json}")
    torch.set_num_threads(1)
    results = []
    for filename in ("config.yaml", "config_wbc.yaml"):
        config_path = DEPLOY / filename
        try:
            results.append(inspect_configuration(config_path))
        except (OSError, RuntimeError, ValueError) as error:
            results.append({"configuration": str(config_path), "passed": False, "error": str(error)})
    report = {
        "schema_version": 1,
        "scope": "asset loading and inference contract only; no physical skill validation",
        "passed": all(result["passed"] for result in results),
        "python": sys.version,
        "python_executable": sys.executable,
        "mujoco": mujoco.__version__,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "torch_cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "torch_threads": torch.get_num_threads(),
        "check_sha256": sha256(Path(__file__)),
        "results": results,
    }
    output = json.dumps(report, indent=2, allow_nan=False)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("x") as destination:
            destination.write(output + "\n")
    print(output)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())