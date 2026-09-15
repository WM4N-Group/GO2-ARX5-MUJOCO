"""Load checksum-bound candidate shards without opening physics snapshots."""

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from ..box_support_scene import BOX_SUPPORT_FAMILIES
from .encoding import encode_example


def read_checked(path, expected=None):
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError(f"Checksum mismatch: {path}")
    return raw, digest


def snapshot_identity(record):
    if record.get("snapshot_id"):
        return f"boundary:{record['snapshot_id']}"
    if record.get("snapshot_sha256"):
        return f"archive:{record['snapshot_sha256']}"
    raise ValueError("Missing skill-boundary identity")


def load_dataset(manifest_paths, *, structured=False):
    samples, metadata, provenance = [], [], []
    spatial = []
    identities, assignments, teachers = set(), {}, set()
    request_counts = Counter()
    for manifest_path in map(Path, manifest_paths):
        raw, digest = read_checked(manifest_path)
        manifest = json.loads(raw)
        if manifest["schema_version"] != 2 or not manifest["complete"] or manifest["split"] != "scene_family_v1":
            raise ValueError("World-model data requires complete family-grouped candidate shards")
        source_raw, _source_hash = read_checked(manifest_path.parent / manifest["source_jsonl"], manifest["source_sha256"])
        sources = [json.loads(line) for line in source_raw.splitlines()]
        candidate_raw, _candidate_hash = read_checked(manifest_path.parent / "candidates.jsonl", manifest["candidates_sha256"])
        candidates = [json.loads(line) for line in candidate_raw.splitlines()]
        if len(candidates) != manifest["records"]:
            raise ValueError("Candidate count differs from manifest")
        provenance.append({"manifest": str(manifest_path.resolve()), "sha256": digest, "source_sha256": manifest["source_sha256"], "candidates_sha256": manifest["candidates_sha256"]})
        for candidate in candidates:
            if candidate["candidate_id"] in identities:
                raise ValueError("Duplicate candidate identity")
            identities.add(candidate["candidate_id"])
            source = sources[candidate["source_record_index"]]
            origin = source["metadata"]
            split = candidate["dataset_split"]
            if split not in ("train", "validation", "test") or candidate["scene_family"] not in BOX_SUPPORT_FAMILIES:
                raise ValueError("Unsupported split or scene family")
            for name in ("dataset_split", "scene_family", "scene_id", "episode_id"):
                if candidate[name] != origin[name]:
                    raise ValueError(f"Candidate/source mismatch: {name}")
            identity = snapshot_identity(candidate)
            if identity != snapshot_identity(origin) or candidate.get("snapshot_sha256") != origin.get("snapshot_sha256"):
                raise ValueError("Candidate/source snapshot mismatch")
            if assignments.setdefault(("snapshot", identity), split) != split:
                raise ValueError("Related data crosses train/validation/test splits")
            for name in ("scene_family", "scene_id", "episode_id", "split_group_id"):
                key = name, candidate.get(name, candidate["scene_family"])
                if assignments.setdefault(key, split) != split:
                    raise ValueError("Related data crosses train/validation/test splits")
            teachers.add(json.dumps({"policies": origin["policy_sha256"], "runtime": candidate["snapshot_code_sha256"]}, sort_keys=True))
            request_counts[(split, candidate["outcome"])] += 1
            if candidate["executed"]:
                transition = candidate["transition"]
                if transition["action"] != candidate["action"] or transition["observation_before"] != source["observation_before"]:
                    raise ValueError("Candidate input differs from recorded skill start")
            encoded = encode_example(candidate, source)
            if encoded is not None:
                samples.append(encoded)
                if structured:
                    from .spatial_encoding import encode_proprio, encode_scene

                    spatial.append({**encode_scene(candidate["transition"]["observation_before"], candidate["action"], origin["scene_parameters"]), "proprio": encode_proprio(origin.get("start_context", {}))})
                action = candidate["action"]
                anchor = action["object_id"] if action["skill"] == 1 else action["support_id"]
                metadata.append({"candidate_id": candidate["candidate_id"], "split": split, "skill": action["skill"], "anchor": anchor, "scene_id": candidate["scene_id"], "episode_id": candidate["episode_id"], "snapshot_id": identity, "snapshot_sha256": candidate.get("snapshot_sha256")})
    if len(teachers) != 1 or not samples:
        raise ValueError("Require executed samples from one frozen skill/runtime contract")
    arrays = {name: np.stack(values) for name, values in zip(("features", "regression", "regression_mask", "binary", "binary_mask"), zip(*samples))}
    if structured:
        arrays.update({name: np.stack([value[name] for value in spatial]) for name in ("objects", "object_mask", "object_ids", "proprio")})
        arrays["bev"] = np.stack([value["bev"] for value in spatial]).astype(np.float16)
    for name in ("features", "regression", "binary"):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"Non-finite training array: {name}")
    return {
        **arrays,
        "split": np.array([row["split"] for row in metadata]),
        "group": np.array([row["skill"] * 2 + int(row["anchor"] == 20) for row in metadata]),
        "metadata": metadata, "provenance": provenance, "teacher": json.loads(next(iter(teachers))),
        "requests": {f"{split}:{outcome}": count for (split, outcome), count in sorted(request_counts.items())},
    }