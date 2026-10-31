"""Run with python -m eval_lab; help and comparisons work without PyTorch."""

import argparse
import importlib.metadata
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from .evaluation import load_cases, score, sha256, summarize
from .runtime import MODEL_ID, ROOT


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")


def source_version():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True,
        ).strip()
        return {"commit": commit, "tracked_changes": bool(dirty)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "tracked_changes": None}


def run(args):
    suite_path = ROOT / "datasets" / f"{args.suite}.jsonl"
    prompt_path = ROOT / "prompts" / f"{args.prompt}.txt"
    cases = load_cases(suite_path)
    prompt = prompt_path.read_text().strip()
    started = datetime.now(timezone.utc)
    stamp = started.strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or ROOT / "results" / "local" / f"{stamp}-{args.suite}-{args.prompt}.json"
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}; choose a new filename")

    from .runtime import LocalModel
    load_started = time.perf_counter()
    runtime = LocalModel()
    load_seconds = time.perf_counter() - load_started
    rows = []
    for case in cases:
        tick = time.perf_counter()
        prediction = runtime.generate(prompt, case["input"], args.max_new_tokens)
        row = score(case, {**prediction, "seconds": time.perf_counter() - tick})
        rows.append(row)
        print(f"{'PASS' if row['correct'] else 'FAIL'} {case['id']}: {row['raw_output']!r}", flush=True)
    report = {
        "schema_version": 1, "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_ID, "revision": runtime.revision,
        "suite": args.suite, "suite_sha256": sha256(suite_path),
        "prompt_name": args.prompt, "prompt": prompt, "prompt_sha256": sha256(prompt_path),
        "settings": {"device": "cpu", "dtype": "float32", "thinking": False,
                     "do_sample": False, "max_new_tokens": args.max_new_tokens},
        "environment": {"python": platform.python_version(), "os": platform.system(),
                        "architecture": platform.machine(), "torch_threads": runtime.torch.get_num_threads(),
                        "packages": {name: importlib.metadata.version(name) for name in
                                     ("torch", "transformers", "huggingface_hub", "safetensors")}},
        "source": source_version(), "model_load_seconds": load_seconds,
        "metrics": summarize(rows), "cases": rows,
    }
    write_report(output, report)
    print(f"Accuracy: {report['metrics']['accuracy']:.1%}; valid labels: {report['metrics']['valid_label_rate']:.1%}")
    print(f"Saved: {output}")


def compare(paths):
    reports = [json.loads(path.read_text()) for path in paths]
    first = reports[0]
    for report in reports:
        if report.get("schema_version") != 1:
            raise ValueError("Comparison requires schema v1 reports; legacy reports are preserved separately")
        for key in ("suite_sha256", "model", "revision", "settings"):
            if report[key] != first[key]:
                raise ValueError(f"Cannot isolate prompt effects: reports differ in {key}")
        if [(row["id"], row["input"], row["expected"]) for row in report["cases"]] != [
            (row["id"], row["input"], row["expected"]) for row in first["cases"]
        ]:
            raise ValueError("Reports contain different or reordered cases")
    print("prompt\tcorrect\taccuracy\tvalid\tmedian seconds")
    for report in reports:
        metrics = report["metrics"]
        print(f"{report['prompt_name']}\t{metrics['correct']}/{metrics['total']}\t"
              f"{metrics['accuracy']:.1%}\t{metrics['valid_label_rate']:.1%}\t{metrics['median_seconds']:.3f}")
    print("Changed outputs (relative to the first report):")
    changes = 0
    for report in reports[1:]:
        for old, new in zip(first["cases"], report["cases"]):
            if old["raw_output"] != new["raw_output"]:
                changes += 1
                print(f"{new['id']} [{report['prompt_name']}]: {old['raw_output']!r} -> {new['raw_output']!r}; expected={new['expected']}")
    if not changes:
        print("None.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Local Qwen ticket-classification evals")
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="Run one dataset with one prompt")
    run_parser.add_argument("--suite", choices=sorted(p.stem for p in (ROOT / "datasets").glob("*.jsonl")), default="smoke")
    run_parser.add_argument("--prompt", choices=sorted(p.stem for p in (ROOT / "prompts").glob("*.txt")), default="clarified")
    run_parser.add_argument("--output", type=Path)
    run_parser.add_argument("--max-new-tokens", type=int, default=16)
    compare_parser = commands.add_parser("compare", help="Compare controlled prompt experiments")
    compare_parser.add_argument("reports", nargs="+", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            if args.max_new_tokens < 1:
                raise ValueError("--max-new-tokens must be positive")
            run(args)
        else:
            compare(args.reports)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
