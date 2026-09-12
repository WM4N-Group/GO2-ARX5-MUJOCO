"""Versioned archives for trusted, self-generated simulator snapshots."""

from __future__ import annotations

from collections import deque
from enum import Enum
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile

import mujoco
import numpy as np
import torch


def _registry() -> dict[str, type]:
    from ..climb_runtime import ClimbRuntime
    from ..complex_course_env import ComplexCourseEnv
    from ..env import BlockedPassageEnv
    from ..locomotion_runtime import LocomotionRuntime
    from ..occupancy import GridConfig, OccupancyGrid
    from ..oracle_planner import OraclePlanner, PlannerConfig
    from ..passage_scene import ParameterizedPassageEnv, PassageScene
    from ..representations import Capability, SkillType
    from ..runtime.executor import ReconfigurableExecutor
    from ..runtime.replanner import OracleReplanner
    from ..runtime.safety import SafetyConfig, SafetyMonitor

    classes = (
        BlockedPassageEnv, ComplexCourseEnv, LocomotionRuntime, ClimbRuntime,
        GridConfig, OccupancyGrid, OraclePlanner, PlannerConfig, Capability,
        SkillType, ReconfigurableExecutor, OracleReplanner, SafetyConfig, SafetyMonitor,
        ParameterizedPassageEnv, PassageScene,
    )
    return {kind.__name__: kind for kind in classes}


def _versions() -> dict[str, str]:
    return {
        "mujoco": mujoco.__version__,
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "byteorder": sys.byteorder,
        "torch_threads": str(torch.get_num_threads()),
    }


def _code_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    files = (
        "env.py", "complex_course_env.py", "locomotion_runtime.py", "climb_runtime.py",
        "representations.py", "occupancy.py", "oracle_planner.py", "runtime/executor.py",
        "runtime/replanner.py", "runtime/safety.py", "skills/base.py", "skills/navigate.py",
        "skills/push.py", "skills/climb.py", "data/transition.py", "data/snapshot.py",
        "data/snapshot_io.py", "data/events.py", "passage_scene.py",
    )
    digest = hashlib.sha256()
    for relative in files:
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update((root / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


class _StateCodec:
    def __init__(self, members: dict[str, bytes]) -> None:
        self.members = members
        self.classes = _registry()
        self.array_count = 0

    def object_state(self, value: object, excluded: tuple[str, ...] = ()) -> dict:
        name = type(value).__name__
        if self.classes.get(name) is not type(value):
            raise TypeError(f"Unsupported snapshot object: {name}")
        attributes = {key: item for key, item in vars(value).items() if key not in excluded}
        return {"kind": "object", "class": name, "attributes": self.encode(attributes)}

    def encode(self, value: object) -> object:
        if isinstance(value, Enum):
            return {"kind": "enum", "class": type(value).__name__, "value": value.value}
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise TypeError("Object arrays are not supported by snapshot archives")
            name = f"arrays/{self.array_count}.npy"
            self.array_count += 1
            buffer = BytesIO()
            np.save(buffer, value, allow_pickle=False)
            self.members[name] = buffer.getvalue()
            return {"kind": "array", "member": name}
        if isinstance(value, np.generic):
            return self.encode(value.item())
        if isinstance(value, np.random.Generator):
            return {"kind": "rng", "state": self.encode(value.bit_generator.state)}
        if isinstance(value, dict):
            if not all(isinstance(key, str) for key in value):
                raise TypeError("Snapshot dictionary keys must be strings")
            return {"kind": "dict", "items": {key: self.encode(item) for key, item in value.items()}}
        if isinstance(value, deque):
            return {"kind": "deque", "maxlen": value.maxlen, "items": [self.encode(item) for item in value]}
        if isinstance(value, (list, tuple, set, frozenset)):
            items = sorted(value, key=repr) if isinstance(value, (set, frozenset)) else value
            return {"kind": type(value).__name__, "items": [self.encode(item) for item in items]}
        if isinstance(value, float) and not math.isfinite(value):
            return {"kind": "float", "value": str(value)}
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        return self.object_state(value)

    def decode(self, value: object) -> object:
        if not isinstance(value, dict):
            return value
        kind = value["kind"]
        if kind == "array":
            return np.load(BytesIO(self.members[value["member"]]), allow_pickle=False)
        if kind == "dict":
            return {key: self.decode(item) for key, item in value["items"].items()}
        if kind == "object":
            object_type = self.classes[value["class"]]
            instance = object_type.__new__(object_type)
            instance.__dict__.update(self.decode(value["attributes"]))
            return instance
        if kind == "enum":
            return self.classes[value["class"]](value["value"])
        if kind == "rng":
            state = self.decode(value["state"])
            generators = {name: getattr(np.random, name) for name in ("PCG64", "PCG64DXSM", "MT19937", "Philox", "SFC64")}
            generator = generators[state["bit_generator"]]()
            generator.state = state
            return np.random.Generator(generator)
        if kind == "deque":
            return deque((self.decode(item) for item in value["items"]), maxlen=value["maxlen"])
        containers = {"list": list, "tuple": tuple, "set": set, "frozenset": frozenset}
        if kind in containers:
            return containers[kind](self.decode(item) for item in value["items"])
        if kind == "float" and value["value"] in ("inf", "-inf", "nan"):
            return float(value["value"])
        raise ValueError(f"Unsupported snapshot state kind: {kind}")


def save_snapshot(snapshot, path: Path | str) -> None:
    executor = snapshot._executor
    members = {
        "model.bin": executor.env.model.__getstate__(),
        "data.bin": executor.env.data.__getstate__(),
    }
    codec = _StateCodec(members)
    metadata = {
        "environment": codec.object_state(executor.env, ("model", "data")),
        "executor": codec.object_state(executor, ("env", "runtime", "climb_runtime")),
        "random_state": codec.encode(snapshot._random_state),
        "previous_skill": codec.encode(snapshot._previous_skill),
    }
    for name, runtime in (("locomotion", executor.runtime), ("climb", executor.climb_runtime)):
        metadata[name] = None if runtime is None else codec.object_state(runtime, ("env", "model", "data", "policy"))
        if runtime is not None:
            policy = BytesIO()
            torch.jit.save(runtime.policy, policy)
            members[f"{name}.pt"] = policy.getvalue()
    members["metadata.json"] = json.dumps(metadata, allow_nan=False, sort_keys=True).encode()
    manifest = {
        "schema_version": 1,
        "versions": _versions(),
        "code_sha256": _code_fingerprint(),
        "members": {name: hashlib.sha256(content).hexdigest() for name, content in members.items()},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".snapshot-")
    os.close(descriptor)
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=3) as archive:
            for name, content in members.items():
                archive.writestr(name, content)
            archive.writestr("manifest.json", json.dumps(manifest, sort_keys=True))
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def load_snapshot(path: Path | str, *, trusted: bool = False):
    from .snapshot import SimulatorSnapshot

    if not trusted:
        raise ValueError("Snapshot contains native state and TorchScript; explicit trusted=True is required")
    with ZipFile(path) as archive:
        entries = archive.infolist()
        if len({entry.filename for entry in entries}) != len(entries):
            raise ValueError("Duplicate snapshot archive members")
        if sum(entry.file_size for entry in entries) > 2 * 1024**3:
            raise ValueError("Snapshot archive exceeds the supported size")
        manifest = json.loads(archive.read("manifest.json"))
        if manifest["schema_version"] != 1 or manifest["versions"] != _versions():
            raise ValueError("Incompatible snapshot schema or runtime versions")
        if manifest["code_sha256"] != _code_fingerprint():
            raise ValueError("Snapshot code fingerprint does not match the current runtime")
        if set(archive.namelist()) != {*manifest["members"], "manifest.json"}:
            raise ValueError("Snapshot manifest does not match archive members")
        members = {}
        for name, expected in manifest["members"].items():
            content = archive.read(name)
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError(f"Snapshot checksum mismatch: {name}")
            members[name] = content
    codec = _StateCodec(members)
    metadata = json.loads(members["metadata.json"])
    model = mujoco.MjModel.__new__(mujoco.MjModel)
    model.__setstate__(members["model.bin"])
    stored_data = mujoco.MjData.__new__(mujoco.MjData)
    stored_data.__setstate__(members["data.bin"])
    if stored_data.qpos.shape != (model.nq,) or stored_data.qvel.shape != (model.nv,):
        raise ValueError("Snapshot model and data dimensions differ")
    data = mujoco.MjData(model)
    mujoco.mj_copyData(data, model, stored_data)
    env = codec.decode(metadata["environment"])
    env.model, env.data = model, data
    executor = codec.decode(metadata["executor"])
    executor.env = env
    for name, attribute in (("locomotion", "runtime"), ("climb", "climb_runtime")):
        runtime = None if metadata[name] is None else codec.decode(metadata[name])
        if runtime is not None:
            runtime.model, runtime.data = model, data
            runtime.policy = torch.jit.load(BytesIO(members[f"{name}.pt"]), map_location="cpu")
            runtime.policy.eval()
            if name == "locomotion":
                runtime.env = env
        setattr(executor, attribute, runtime)
    random_state = codec.decode(metadata["random_state"])
    previous_skill = codec.decode(metadata["previous_skill"])
    return SimulatorSnapshot(executor, random_state, previous_skill)