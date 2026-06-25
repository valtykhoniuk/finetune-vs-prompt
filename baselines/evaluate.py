from collections import Counter

def compute_metrics(predictions: list[dict]) -> dict:
    """
    predictions: [{"text: "...", "gold": "billing", "pred": "billing"}, ...]
    """
    total = len(predictions)
    correct = sum(1 for p in predictions if p["gold"] == p["pred"])
    accuracy = correct / total if total else 0.0

    by_label = Counter(p["gold"] for p in predictions)
    correct_by_label = Counter(
        p["gold"] for p in predictions if p["gold"] == p["pred"]
    )

    per_label = {}
    for label in by_label:
        n = by_label[label]
        c = correct_by_label.get(label, 0)
        per_label[label] = {"correct": c, "total": n, "accuracy": c/n}

    return {
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "per_label": per_label
    }
