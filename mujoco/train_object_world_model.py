"""Compare residual MLP and object/BEV Transformer on identical grouped data."""

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from reconfigurable_navigation.world_model.dataset import load_dataset
from reconfigurable_navigation.world_model.evaluation import conditional_baseline, metrics, selection_metrics
from reconfigurable_navigation.world_model.object_model import ObjectWorldModel
from reconfigurable_navigation.world_model.spatial_encoding import BEV_CHANNELS, BEV_RESOLUTION, BEV_SIZE, OBJECT_TOKEN_NAMES, PROPRIO_SIZE, SPATIAL_SCHEMA


INPUTS = ("features", "bev", "objects", "object_mask", "object_ids", "proprio")
TARGETS = ("regression", "regression_mask", "binary", "binary_mask")


def train_one(dataset, indices, args, architecture, output):
    torch.manual_seed(args.seed)
    generator = np.random.default_rng(args.seed)
    model = ObjectWorldModel(args.width, args.layers, args.heads, architecture)
    model.set_normalization(dataset)
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    tensors = {name: torch.from_numpy(dataset[name]) for name in (*INPUTS, *TARGETS)}

    def inputs(selected):
        return tuple(None if architecture == "residual_mlp" and name in ("bev", "objects", "object_mask", "object_ids") else tensors[name][selected].to(args.device) for name in INPUTS)

    def batch_loss(selected):
        return model.loss(inputs(selected), *(tensors[name][selected].to(args.device) for name in TARGETS))

    def validation_loss():
        model.eval()
        losses = []
        with torch.inference_mode():
            for offset in range(0, len(indices["validation"]), args.batch_size):
                selected = indices["validation"][offset:offset + args.batch_size]
                losses.append(float(batch_loss(selected)) * len(selected))
        return sum(losses) / len(indices["validation"])

    initial_loss = validation_loss()
    best_loss, best_epoch, best_state = initial_loss, 0, deepcopy(model.state_dict())
    history = [{"epoch": 0, "validation_loss": initial_loss}]
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = generator.permutation(indices["train"])
        losses = []
        for offset in range(0, len(order), args.batch_size):
            selected = order[offset:offset + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(selected)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            losses.append(float(loss.detach()) * len(selected))
        score = validation_loss()
        if not np.isfinite(score):
            raise RuntimeError("Non-finite validation loss")
        history.append({"epoch": epoch, "train_loss": sum(losses) / len(order), "validation_loss": score})
        if score < best_loss - 1e-8:
            best_loss, best_epoch, best_state = score, epoch, deepcopy(model.state_dict())
        if epoch == 1 or epoch % 20 == 0:
            print(f"OBJECT_MODEL architecture={architecture} epoch={epoch} train={history[-1]['train_loss']:.4f} validation={score:.4f} best={best_epoch}", flush=True)
        if epoch - best_epoch >= args.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    regression, probabilities = [], []
    with torch.inference_mode():
        for offset in range(0, len(dataset["features"]), args.batch_size):
            selected = np.arange(offset, min(offset + args.batch_size, len(dataset["features"])))
            values, logits = model(*inputs(selected))
            regression.append(values.cpu().numpy())
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    regression, probabilities = np.concatenate(regression), np.concatenate(probabilities)
    report = {
        "architecture": architecture, "config": model.config, "best_epoch": best_epoch,
        "learned_residual_used": best_epoch > 0, "best_validation_loss": best_loss,
        "initial_prior_validation_loss": initial_loss, "history": history,
        "training_seconds": time.perf_counter() - started,
        "metrics": {split: metrics(dataset, selected, regression, probabilities) for split, selected in indices.items()},
        "selection": {split: selection_metrics(dataset, selected, regression, probabilities) for split, selected in indices.items()},
    }
    output.mkdir()
    metadata = {"scope": "offline_single_box_support", "schema": SPATIAL_SCHEMA, "seed": args.seed, "best_epoch": best_epoch, "teacher": dataset["teacher"], "data_provenance": dataset["provenance"], "scene_families_grouped": True, "proprio_size": PROPRIO_SIZE, "status": "offline_comparison_only"}
    model.save(output / "model.pt", metadata)
    restored, _metadata = ObjectWorldModel.load(output / "model.pt")
    restored.to(args.device).eval()
    with torch.inference_mode():
        for actual, expected in zip(restored(*inputs(np.arange(min(8, len(dataset["features"]))))), model(*inputs(np.arange(min(8, len(dataset["features"])))))):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    report["checkpoint_sha256"] = hashlib.sha256((output / "model.pt").read_bytes()).hexdigest()
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    with (output / "predictions.jsonl").open("x") as stream:
        for index, row in enumerate(dataset["metadata"]):
            stream.write(json.dumps({**row, "regression": regression[index].tolist(), "probabilities": probabilities[index].tolist(), "target": dataset["regression"][index].tolist(), "binary_target": dataset["binary"][index].tolist(), "regression_mask": dataset["regression_mask"][index].tolist(), "binary_mask": dataset["binary_mask"][index].tolist()}, allow_nan=False) + "\n")
    del restored, model, optimizer, best_state
    if str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--architectures", nargs="+", choices=("residual_mlp", "object_transformer"), default=["residual_mlp", "object_transformer"])
    args = parser.parse_args()
    if min(args.width, args.layers, args.heads, args.epochs, args.patience, args.batch_size) < 1 or args.seed < 0 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Invalid training settings")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            parser.error("CUDA is unavailable in this environment")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    dataset = load_dataset(args.manifest, structured=True)
    indices = {split: np.flatnonzero(dataset["split"] == split) for split in ("train", "validation", "test")}
    if any(not len(selected) for selected in indices.values()):
        parser.error("Nonempty family-grouped training, validation and test sets are required")
    if not dataset["proprio"][:, -1].all():
        parser.error("This comparison requires recorded start-of-skill proprioception")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    baseline, baseline_probabilities = conditional_baseline(dataset)
    report = {
        "schema_version": 1, "input_schema": SPATIAL_SCHEMA, "seed": args.seed, "device": args.device,
        "bev_shape": [8, BEV_SIZE, BEV_SIZE], "bev_resolution": BEV_RESOLUTION, "bev_channels": BEV_CHANNELS,
        "object_token_names": OBJECT_TOKEN_NAMES, "proprio_size": PROPRIO_SIZE,
        "data_provenance": dataset["provenance"], "teacher": dataset["teacher"], "requests": dataset["requests"],
        "split_counts": {split: len(selected) for split, selected in indices.items()},
        "numpy_version": np.__version__, "torch_version": str(torch.__version__),
        "baseline": {split: metrics(dataset, selected, baseline, baseline_probabilities) for split, selected in indices.items()},
        "baseline_selection": {split: selection_metrics(dataset, selected, baseline, baseline_probabilities) for split, selected in indices.items()},
        "models": {},
        "code_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (Path(__file__), *sorted((Path(__file__).parent / "reconfigurable_navigation/world_model").glob("*.py")))},
        "limitations": ["privileged yaw-box BEV, not reconstructed RGB-D", "ground-plane completeness assumed within the fixed support scene", "conditional statistical priors and proprioception are shared with the residual MLP control", "single trained object model, not an ensemble", "offline one-skill comparison, no learned closed-loop control"],
    }
    for architecture in args.architectures:
        report["models"][architecture] = train_one(dataset, indices, args, architecture, args.output_dir / architecture)
    (args.output_dir / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"split_counts": report["split_counts"], "baseline_test_selection": report["baseline_selection"]["test"], "models": {name: {"best_epoch": result["best_epoch"], "test": result["metrics"]["test"], "test_selection": result["selection"]["test"]} for name, result in report["models"].items()}}, indent=2))


if __name__ == "__main__":
    main()