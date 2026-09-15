"""Masked physical-unit metrics and train-only conditional baselines."""

import numpy as np

from .encoding import BINARY_NAMES


def conditional_baseline(dataset):
    train = dataset["split"] == "train"
    regression = np.zeros_like(dataset["regression"])
    probabilities = np.zeros_like(dataset["binary"])
    for group in np.unique(dataset["group"]):
        selected = train & (dataset["group"] == group)
        if not selected.any():
            selected = train
        target, mask = dataset["regression"][selected], dataset["regression_mask"][selected]
        mean = (target * mask).sum(axis=0) / np.maximum(mask.sum(axis=0), 1)
        labels, valid = dataset["binary"][selected], dataset["binary_mask"][selected]
        probability = ((labels * valid).sum(axis=0) + 0.5) / (valid.sum(axis=0) + 1.0)
        regression[dataset["group"] == group] = mean
        probabilities[dataset["group"] == group] = probability
    return regression, probabilities


def _classification(labels, probabilities):
    count = len(labels)
    if not count:
        return {"count": 0, "positives": 0, "brier": None, "log_loss": None, "accuracy": None, "auroc": None, "ece": None}
    probabilities = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    positive, negative = probabilities[labels == 1], probabilities[labels == 0]
    auroc = None if not len(positive) or not len(negative) else float(((positive[:, None] > negative).astype(float) + 0.5 * (positive[:, None] == negative)).mean())
    calibration = 0.0
    bins = np.minimum((probabilities * 10).astype(int), 9)
    for index in range(10):
        selected = bins == index
        if selected.any():
            calibration += float(selected.mean() * abs(labels[selected].mean() - probabilities[selected].mean()))
    return {
        "count": count, "positives": int(labels.sum()),
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "log_loss": float(-np.mean(labels * np.log(probabilities) + (1.0 - labels) * np.log1p(-probabilities))),
        "accuracy": float(np.mean((probabilities >= 0.5) == labels)), "auroc": auroc, "ece": calibration,
    }


def metrics(dataset, indices, predicted, probabilities):
    target, mask = dataset["regression"][indices], dataset["regression_mask"][indices]
    predicted = predicted[indices]
    result = {"samples": len(indices)}
    for name, columns in (("robot", slice(0, 3)), ("box", slice(5, 8))):
        valid = mask[:, columns].all(axis=1)
        error = np.linalg.norm(predicted[valid, columns] - target[valid, columns], axis=1)
        result[f"{name}_position_rmse_m"] = float(np.sqrt(np.mean(error ** 2))) if len(error) else None
        result[f"{name}_position_p95_m"] = float(np.quantile(error, 0.95)) if len(error) else None
    for name, column in (("robot", 3), ("box", 8)):
        valid = mask[:, column:column + 2].all(axis=1)
        delta = np.arctan2(predicted[valid, column], predicted[valid, column + 1]) - np.arctan2(target[valid, column], target[valid, column + 1])
        result[f"{name}_yaw_mae_rad"] = float(np.mean(np.abs(np.arctan2(np.sin(delta), np.cos(delta))))) if len(delta) else None
    for name, column in (("skill", 10), ("total_attempt", 11)):
        valid = mask[:, column]
        actual = np.expm1(target[valid, column].astype(np.float64))
        estimate = np.expm1(np.clip(predicted[valid, column].astype(np.float64), 0, 80))
        result[f"{name}_seconds_mae"] = float(np.mean(np.abs(estimate - actual))) if len(actual) else None
    result["classification"] = {}
    for column, name in enumerate(BINARY_NAMES):
        valid = dataset["binary_mask"][indices, column]
        result["classification"][name] = _classification(dataset["binary"][indices, column][valid], probabilities[indices, column][valid])
    return result


def selection_metrics(dataset, indices, predicted, probabilities):
    groups = {}
    for index in indices:
        metadata = dataset["metadata"][index]
        groups.setdefault(metadata.get("snapshot_id") or metadata["snapshot_sha256"], []).append(index)
    selected_success, best_success, regrets = [], [], []
    for members in groups.values():
        members = np.asarray(members)
        if len(members) < 2 or not dataset["binary_mask"][members, 2].all() or not dataset["regression_mask"][members, 11].all():
            continue
        costs = np.expm1(np.clip(predicted[members, 11].astype(np.float64), 0, 80))
        scores = probabilities[members, 2] - probabilities[members, 1] - 0.001 * costs
        chosen = members[int(np.argmax(scores))]
        actual = dataset["binary"][members, 2]
        succeeded = float(dataset["binary"][chosen, 2])
        selected_success.append(succeeded)
        best_success.append(float(actual.max()))
        if succeeded:
            best_cost = np.expm1(dataset["regression"][members[actual == 1], 11].astype(np.float64)).min()
            regrets.append(float(np.expm1(dataset["regression"][chosen, 11]) - best_cost))
    return {
        "evaluated_starts": len(selected_success),
        "selected_task_success_rate": float(np.mean(selected_success)) if selected_success else None,
        "best_observed_candidate_success_rate": float(np.mean(best_success)) if best_success else None,
        "successful_selection_cost_regret_seconds": float(np.mean(regrets)) if regrets else None,
        "score": "task_probability - collision_probability - 0.001 * attempt_seconds",
    }