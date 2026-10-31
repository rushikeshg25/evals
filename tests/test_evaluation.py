"""Fast correctness checks. No model download, API key, or third-party library."""

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from eval_lab.__main__ import compare, run, write_report
from eval_lab.evaluation import LABELS, load_cases, score, sha256, summarize
from eval_lab.runtime import ROOT


def case(label="billing", identifier="case-1"):
    return {"id": identifier, "category": "direct", "input": "A duplicate charge", "expected": label}


def row(label, output, identifier="case-1"):
    return score(case(label, identifier), {"raw_output": output, "seconds": 0.1})


class ScoringTests(unittest.TestCase):
    def test_normalizes_only_case_and_outer_whitespace(self):
        self.assertTrue(row("billing", " BILLING\n")["correct"])
        for output in ("billing.", "The label is billing", "", '"billing"', "billing login"):
            with self.subTest(output=output):
                self.assertFalse(row("billing", output)["valid_label"])
                self.assertFalse(row("billing", output)["correct"])

    def test_constant_prediction_is_not_success(self):
        metrics = summarize([row(label, "other", label) for label in LABELS])
        self.assertAlmostEqual(metrics["accuracy"], 1 / 3)
        self.assertEqual(metrics["valid_label_rate"], 1)
        self.assertEqual(metrics["per_label_recall"], {"billing": 0, "login": 0, "other": 1})
        self.assertEqual(metrics["prediction_counts"], {"other": 3})

    def test_invalid_answers_count_as_failures(self):
        metrics = summarize([row("billing", "billing"), row("login", "a poem", "case-2")])
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["valid_label_rate"], 0.5)
        self.assertEqual(metrics["confusion_matrix"]["login"]["invalid"], 1)
        self.assertIsNone(metrics["per_label_recall"]["other"])

    def test_category_metrics_and_truncation(self):
        first = row("billing", "billing")
        second = {**row("login", "other", "case-2"), "category": "negation", "hit_token_limit": True}
        metrics = summarize([first, second])
        self.assertEqual(metrics["per_category"]["direct"]["accuracy"], 1)
        self.assertEqual(metrics["per_category"]["negation"]["accuracy"], 0)
        self.assertEqual(metrics["token_limit_count"], 1)

    def test_empty_run_is_rejected(self):
        with self.assertRaises(ValueError):
            summarize([])


class DatasetTests(unittest.TestCase):
    def load(self, cases):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            path.write_text("".join(json.dumps(value) + "\n" for value in cases))
            return load_cases(path)

    def test_rejects_invalid_datasets(self):
        invalid = [[], [[]], [None], [case("unknown")], [{**case(), "input": " "}],
                   [case(), case(identifier="case-2")],
                   [case(), {**case(), "input": "different"}]]
        for cases in invalid:
            with self.subTest(cases=cases), self.assertRaises(ValueError):
                self.load(cases)

    def test_normalized_duplicates_are_rejected(self):
        with self.assertRaises(ValueError):
            self.load([case(), {**case(identifier="case-2"), "input": " A DUPLICATE  CHARGE "}])

    def test_suites_are_balanced_and_disjoint(self):
        texts = set()
        identifiers = set()
        for name, count in (("smoke", 6), ("development", 36), ("holdout", 12)):
            cases = load_cases(ROOT / "datasets" / f"{name}.jsonl")
            self.assertEqual(len(cases), count)
            self.assertEqual(Counter(item["expected"] for item in cases), {label: count // 3 for label in LABELS})
            for item in cases:
                text = " ".join(item["input"].lower().split())
                self.assertNotIn(text, texts)
                self.assertNotIn(item["id"], identifiers)
                texts.add(text)
                identifiers.add(item["id"])
        examples = (ROOT / "prompts/few-shot.txt").read_text().splitlines()
        for line in examples:
            if line.startswith("Ticket: "):
                self.assertNotIn(" ".join(line[8:].lower().split()), texts)

    def test_original_smoke_inputs_are_preserved(self):
        original = json.loads((ROOT / "results/legacy/baseline.json").read_text())
        current = load_cases(ROOT / "datasets/smoke.jsonl")
        self.assertEqual([(r["input"], r["expected"]) for r in original["cases"]],
                         [(r["input"], r["expected"]) for r in current])


class RunnerTests(unittest.TestCase):
    def test_never_overwrites_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            write_report(path, {"original": True})
            with self.assertRaises(FileExistsError):
                write_report(path, {"replacement": True})
            self.assertEqual(json.loads(path.read_text()), {"original": True})

    def test_existing_output_is_rejected_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            output.write_text("existing")
            args = SimpleNamespace(suite="smoke", prompt="baseline", output=output, max_new_tokens=16)
            with patch("eval_lab.runtime.LocalModel") as runtime:
                with self.assertRaises(FileExistsError):
                    run(args)
                runtime.assert_not_called()

    def test_runner_records_all_cases_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            args = SimpleNamespace(suite="smoke", prompt="baseline", output=output, max_new_tokens=16)
            with patch("eval_lab.runtime.LocalModel") as runtime, \
                 patch("eval_lab.__main__.importlib.metadata.version", return_value="test-version"), \
                 patch("eval_lab.__main__.source_version", return_value={"commit": "test-commit"}), \
                 contextlib.redirect_stdout(io.StringIO()):
                runtime.return_value.revision = "test-revision"
                runtime.return_value.torch.get_num_threads.return_value = 1
                runtime.return_value.generate.return_value = {
                    "raw_output": "other", "input_tokens": 10,
                    "output_tokens": 2, "hit_token_limit": False,
                }
                run(args)
            report = json.loads(output.read_text())
            self.assertEqual(len(report["cases"]), 6)
            self.assertEqual(report["metrics"]["correct"], 2)
            self.assertEqual(report["revision"], "test-revision")
            self.assertEqual(report["suite_sha256"], sha256(ROOT / "datasets/smoke.jsonl"))
            self.assertEqual(report["prompt"], (ROOT / "prompts/baseline.txt").read_text().strip())
            self.assertEqual(report["settings"]["max_new_tokens"], 16)

    def test_help_works_without_model_libraries_or_project_cwd(self):
        script = "import sys; sys.path.insert(0, sys.argv[1]); from eval_lab.__main__ import main; main(['--help'])"
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-B", "-S", "-c", script, str(ROOT)],
                                    cwd=directory, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("compare", result.stdout)
        self.assertNotIn("Loading", result.stdout)


class ComparisonTests(unittest.TestCase):
    def report(self, output="other"):
        rows = [row("billing", output)]
        return {"schema_version": 1, "suite_sha256": "same-data", "model": "same-model",
                "revision": "same-revision", "settings": {"max_new_tokens": 16},
                "prompt_name": "test", "metrics": summarize(rows), "cases": rows}

    def compare_reports(self, reports):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for i, report in enumerate(reports):
                path = Path(directory) / f"{i}.json"
                path.write_text(json.dumps(report))
                paths.append(path)
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                compare(paths)
            return captured.getvalue()

    def test_reports_changed_answers(self):
        output = self.compare_reports([self.report(), self.report("billing")])
        self.assertIn("'other' -> 'billing'", output)

    def test_rejects_different_experiment_conditions(self):
        for field in ("suite_sha256", "model", "revision", "settings"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.compare_reports([self.report(), {**self.report(), field: "different"}])

    def test_rejects_different_cases_despite_matching_declared_hash(self):
        changed = self.report()
        changed["cases"][0]["input"] = "A different ticket"
        with self.assertRaises(ValueError):
            self.compare_reports([self.report(), changed])

    def test_rejects_legacy_schema(self):
        with self.assertRaises(ValueError):
            self.compare_reports([{"accuracy": 1}])


if __name__ == "__main__":
    unittest.main()
