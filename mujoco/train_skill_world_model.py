"""Train and evaluate the first privileged box-support world-model baseline."""

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from reconfigurable_navigation.world_model.dataset import load_dataset
from reconfigurable_navigation.world_model.encoding import BINARY_NAMES, FEATURE_NAMES, REGRESSION_NAMES, SCHEMA
from reconfigurable_navigation.world_model.evaluation import conditional_baseline, metrics, selection_metrics
from reconfigurable_navigation.world_model.model import SkillWorldModel, fit_normalization


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.hidden_size, args.batch_size) < 1 or args.seed < 0:
        parser.error("Invalid training settings")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    dataset = load_dataset(args.manifest)
    indices = {split: np.flatnonzero(dataset["split"] == split) for split in ("train", "validation", "test")}
    if any(not len(selected) for selected in indices.values()):
        parser.error("Nonempty train, validation and test groups are required")
    if {dataset["metadata"][index]["skill"] for index in indices["train"]} != {0, 1, 2}:
        parser.error("This baseline requires all three skills in the training group")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    model = SkillWorldModel(args.hidden_size)
    model.set_normalization(fit_normalization(dataset))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    tensors = {name: torch.from_numpy(dataset[name]) for name in ("features", "regression", "regression_mask", "binary", "binary_mask")}

    def loss(selected):
        return model.loss(*(values[selected] for values in tensors.values()))

    best_loss, best_epoch, best_state = float("inf"), 0, None
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = indices["train"][torch.randperm(len(indices["train"])).numpy()]
        train_losses = []
        for offset in range(0, len(order), args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            current = loss(order[offset:offset + args.batch_size])
            current.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            train_losses.append(float(current.detach()))
        model.eval()
        with torch.inference_mode():
            validation_loss = float(loss(indices["validation"]))
        if not np.isfinite(validation_loss):
            raise RuntimeError("Non-finite validation loss")
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "validation_loss": validation_loss})
        if validation_loss < best_loss - 1e-8:
            best_loss, best_epoch, best_state = validation_loss, epoch, deepcopy(model.state_dict())
        if epoch == 1 or epoch % 25 == 0:
            print(f"WORLD_MODEL epoch={epoch} train={history[-1]['train_loss']:.4f} validation={validation_loss:.4f} best_epoch={best_epoch}", flush=True)
        if epoch - best_epoch >= args.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        predicted, logits = model(tensors["features"])
        predicted, probabilities = predicted.numpy(), torch.sigmoid(logits).numpy()
    baseline, baseline_probabilities = conditional_baseline(dataset)
    report = {
        "schema_version": 1, "model_schema": SCHEMA,
        "architecture": {"type": "MLP", "hidden_sizes": [args.hidden_size, args.hidden_size], "input_size": len(FEATURE_NAMES), "regression_heads": REGRESSION_NAMES, "binary_heads": BINARY_NAMES},
        "seed": args.seed, "best_epoch": best_epoch, "best_validation_loss": best_loss,
        "training_seconds": time.perf_counter() - started, "history": history,
        "data_provenance": dataset["provenance"], "teacher": dataset["teacher"], "requests": dataset["requests"],
        "split_counts": {split: len(selected) for split, selected in indices.items()},
        "skill_counts": dict(Counter(str(row["skill"]) for row in dataset["metadata"])),
        "torch_version": str(torch.__version__), "numpy_version": np.__version__,
        "code_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (Path(__file__), *sorted((Path(__file__).parent / "reconfigurable_navigation/world_model").glob("*.py")))},
        "metrics": {}, "selection": {}, "per_skill": {},
        "limitations": ["small selected single-box layout families", "pose-transition MLP only; no BEV, object Transformer or ensemble", "no proof of unreachability; task head predicts the specified rule suffix", "control-boundary collision labels, not continuous safety guarantees", "offline scoring only; production executor remains rule-based"],
    }
    for split, selected in indices.items():
        report["metrics"][split] = {"model": metrics(dataset, selected, predicted, probabilities), "baseline": metrics(dataset, selected, baseline, baseline_probabilities)}
        report["selection"][split] = {"model": selection_metrics(dataset, selected, predicted, probabilities), "baseline": selection_metrics(dataset, selected, baseline, baseline_probabilities)}
        report["per_skill"][split] = {}
        for skill, name in enumerate(("NAV", "PUSH", "CLIMB")):
            subset = np.array([index for index in selected if dataset["metadata"][index]["skill"] == skill], dtype=int)
            report["per_skill"][split][name] = {"model": metrics(dataset, subset, predicted, probabilities), "baseline": metrics(dataset, subset, baseline, baseline_probabilities)}
    metadata = {key: report[key] for key in ("model_schema", "architecture", "seed", "best_epoch", "data_provenance", "teacher", "code_sha256", "limitations")}
    checkpoint = args.output_dir / "model.pt"
    model.save(checkpoint, metadata)
    restored, _metadata = SkillWorldModel.load(checkpoint)
    with torch.inference_mode():
        for actual, expected in zip(restored(tensors["features"]), model(tensors["features"])):
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    report["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with (args.output_dir / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    with (args.output_dir / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        for index, row in enumerate(dataset["metadata"]):
            payload = {**row, "regression_target": dataset["regression"][index].tolist(), "regression_mask": dataset["regression_mask"][index].tolist(), "binary_target": dataset["binary"][index].tolist(), "binary_mask": dataset["binary_mask"][index].tolist(), "model_regression": predicted[index].tolist(), "model_probabilities": probabilities[index].tolist(), "baseline_regression": baseline[index].tolist(), "baseline_probabilities": baseline_probabilities[index].tolist()}
            stream.write(json.dumps(payload, allow_nan=False) + "\n")
    print(json.dumps({"best_epoch": best_epoch, "test": report["metrics"]["test"], "selection_test": report["selection"]["test"]}, indent=2))


if __name__ == "__main__":
    main()