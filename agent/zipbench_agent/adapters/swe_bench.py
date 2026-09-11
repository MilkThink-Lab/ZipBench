"""Adapter for the SWE-bench evaluation harness (swebench package).

Recognised result inputs:

1. The harness summary report (``<model>.<run_id>.json``) with
   ``resolved_ids`` / ``unresolved_ids`` / ``error_ids`` / ``empty_patch_ids``.
2. A per-instance conversion ``{"results": [{"instance_id", "correct"}, ...]}``
   (the format used by ZipBench model records).
3. A ``logs/run_evaluation/<run_id>/<model>/`` directory — merges every
   ``<instance_id>/report.json`` (``{instance_id: {"resolved": bool}}``).

``error_ids`` instances score 0.0, matching the official resolved rate
(an evaluation error counts as unresolved). They are *also* put in the
ParseResult's ``errored`` set — the one sanctioned scores/errored overlap
(see base.ParseResult): the ids stay scored, never missing, and the overlap
only surfaces them as "counted 0 despite erroring" in reports.
"""

import json
from pathlib import Path

from .base import Adapter, ParseResult, ResultParseError


class SweBenchAdapter(Adapter):

    def __init__(self, name):
        self.name = name

    def task_args(self, question_ids):
        return ["--instance_ids"] + list(question_ids)

    def parse_results(self, path):
        path = Path(path)
        if path.is_dir():
            return self._parse_log_dir(path)
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            raise ResultParseError(f"cannot read {path}: {e}")

        if isinstance(data, dict) and "resolved_ids" in data:
            return self._parse_summary_report(data)
        if isinstance(data, dict) and isinstance(data.get("results"), list):
            return ParseResult(scores=self._parse_per_instance(data, path))
        # a single (or merged) instance-level report.json
        if isinstance(data, dict) and data and all(
                isinstance(v, dict) and "resolved" in v for v in data.values()):
            return ParseResult(
                scores={iid: float(bool(v["resolved"]))
                        for iid, v in data.items()})
        raise ResultParseError(
            f"{path} is not a recognised swebench result: expected a harness "
            f"summary report (resolved_ids), a per-instance conversion "
            f"(results list), or an instance report.json")

    @staticmethod
    def _parse_summary_report(data):
        scores = {iid: 1.0 for iid in data["resolved_ids"]}
        for key in ("unresolved_ids", "error_ids", "empty_patch_ids"):
            for iid in data.get(key, []):
                scores.setdefault(iid, 0.0)
        # error_ids stay scored (0.0, the official convention) *and* go into
        # errored, purely as diagnostics — the sanctioned overlap.
        errored = {iid for iid in data.get("error_ids", []) if iid in scores}
        return ParseResult(scores=scores, errored=errored)

    @staticmethod
    def _parse_per_instance(data, path):
        scores = {}
        for row in data["results"]:
            try:
                scores[row["instance_id"]] = float(row["correct"])
            except (KeyError, TypeError) as e:
                raise ResultParseError(f"bad results row in {path}: {row!r} ({e})")
        return scores

    def _parse_log_dir(self, path):
        scores = {}
        for report in sorted(path.glob("*/report.json")):
            try:
                data = json.loads(report.read_text())
            except (OSError, json.JSONDecodeError) as e:
                raise ResultParseError(f"cannot read {report}: {e}")
            for iid, v in data.items():
                scores[iid] = float(bool(v.get("resolved")))
        if not scores:
            raise ResultParseError(
                f"{path} contains no <instance_id>/report.json files")
        return ParseResult(scores=scores)
