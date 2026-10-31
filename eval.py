import json
import sys
import time
from pathlib import Path

from model import MODEL_ID, REVISION, generate

PROMPT = (
    "Classify the user's current support problem. "
    "billing: charges, invoices, payments, or refunds. "
    "login: signing in, passwords, or account access. "
    "other: everything else. "
    "Classify the problem, not incidental words. "
    "Return exactly one label: billing, login, or other. "
    "Do not include an explanation."
)

CASES = [
    ("I was charged twice.", "billing"),
    ("My password reset link expired.", "login"),
    ("Please add dark mode.", "other"),
    ("I can log in, but my invoice is wrong.", "billing"),
    ("I paid my bill, but now I cannot sign in.", "login"),
    ("Where can I change the interface language?", "other"),
]

output_path = Path(
    sys.argv[1] if len(sys.argv) > 1 else "baseline.json"
)

# Avoid replacing a previous experiment accidentally.
if output_path.exists():
    raise SystemExit(f"Choose a new output filename: {output_path}")

rows = []

for text, expected in CASES:
    started = time.perf_counter()
    raw = generate(PROMPT, text, max_new_tokens=16)
    elapsed = time.perf_counter() - started

    predicted = raw.strip().lower()
    correct = predicted == expected
    valid = predicted in {"billing", "login", "other"}

    rows.append({
        "input": text,
        "expected": expected,
        "raw_output": raw,
        "correct": correct,
        "valid_label": valid,
        "seconds": elapsed,
    })

    print(
        f"{'PASS' if correct else 'FAIL'} "
        f"expected={expected} actual={raw!r}",
        flush=True,
    )

report = {
    "model": MODEL_ID,
    "revision": REVISION,
    "prompt": PROMPT,
    "settings": {
        "device": "cpu",
        "dtype": "float32",
        "thinking": False,
        "do_sample": False,
        "max_new_tokens": 16,
    },
    "accuracy": sum(row["correct"] for row in rows) / len(rows),
    "valid_label_rate": (
        sum(row["valid_label"] for row in rows) / len(rows)
    ),
    "cases": rows,
}

output_path.write_text(json.dumps(report, indent=2))
print(f"Accuracy: {report['accuracy']:.1%}")
print(f"Valid labels: {report['valid_label_rate']:.1%}")
print(f"Saved: {output_path}")