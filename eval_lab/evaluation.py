"""Dataset validation and scoring; no model libraries are required."""

import hashlib
import json
from collections import Counter
from statistics import mean, median

LABELS = ("billing", "login", "other")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cases(path):
    cases = []
    seen = set()
    texts = set()
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        case = json.loads(line)
        for key in ("id", "category", "input", "expected"):
            if not isinstance(case.get(key), str) or not case[key].strip():
                raise ValueError(f"{path.name}:{line_number}: missing/nonempty string required: {key}")
        if case["expected"] not in LABELS:
            raise ValueError(f"{case['id']}: unknown expected label")
        normalized_text = " ".join(case["input"].lower().split())
        if case["id"] in seen or normalized_text in texts:
            raise ValueError(f"{path.name}:{line_number}: duplicate case ID or input")
        seen.add(case["id"])
        texts.add(normalized_text)
        cases.append(case)
    if not cases:
        raise ValueError(f"{path.name}: dataset must not be empty")
    return cases


def score(case, prediction):
    raw = prediction["raw_output"]
    label = raw.strip().lower()
    # Only casing and outer whitespace are normalized. Extra prose stays invalid.
    return {
        **case, **prediction, "predicted": label,
        "correct": label == case["expected"], "valid_label": label in LABELS,
    }


def summarize(rows):
    if not rows:
        raise ValueError("Cannot score an empty run")
    confusion = {label: {prediction: 0 for prediction in (*LABELS, "invalid")} for label in LABELS}
    per_category = {}
    for row in rows:
        prediction = row["predicted"] if row["valid_label"] else "invalid"
        confusion[row["expected"]][prediction] += 1
        group = per_category.setdefault(row["category"], {"correct": 0, "total": 0})
        group["correct"] += int(row["correct"])
        group["total"] += 1
    for group in per_category.values():
        group["accuracy"] = group["correct"] / group["total"]
    recalls = {}
    for label, counts in confusion.items():
        total = sum(counts.values())
        recalls[label] = counts[label] / total if total else None
    measured_recalls = [value for value in recalls.values() if value is not None]
    return {
        "total": len(rows), "correct": sum(row["correct"] for row in rows),
        "accuracy": mean(row["correct"] for row in rows),
        "valid_label_rate": mean(row["valid_label"] for row in rows),
        "macro_recall": mean(measured_recalls), "per_label_recall": recalls,
        "per_category": per_category, "confusion_matrix": confusion,
        "prediction_counts": dict(Counter(row["predicted"] for row in rows)),
        "token_limit_count": sum(row.get("hit_token_limit", False) for row in rows),
        "mean_seconds": mean(row["seconds"] for row in rows),
        "median_seconds": median(row["seconds"] for row in rows),
    }
