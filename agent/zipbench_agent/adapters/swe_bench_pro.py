"""Adapter for the SWE-Bench Pro evaluation harness (scaleapi/SWE-bench_Pro-os,
``swe_bench_pro_eval.py``).

The harness has no task-filter flag: it evaluates exactly the instances listed
in the predictions JSON (``[{"instance_id", "patch", "prefix"}, ...]``), so
``task_args`` raises — restrict the predictions file to the subset ids from
``tasks swe_bench_pro --format ids`` instead.

Recognised result inputs:

1. The harness output ``{output_dir}/eval_results.json`` — a flat
   ``{instance_id: true/false}`` mapping (or a directory containing it).
2. A per-instance conversion ``{"results": [{"instance_id", "correct"}, ...]}``
   (the format used by ZipBench model records).

The per-instance ``{uid}/{prefix}_output.json`` files are *not* parsed: they
hold raw test statuses, and deciding pass/fail from them needs the dataset's
FAIL_TO_PASS / PASS_TO_PASS expectations, which this dependency-free package
does not ship. Use the harness's own ``eval_results.json``.

Neither accepted format leaves any trace of evaluation errors (the flat bool
mapping just omits the instance), so the ParseResult's ``errored`` set is
always empty here — missing instances all report as absent.
"""

import json
from pathlib import Path

from .base import Adapter, ParseResult, ResultParseError, TaskArgsUnsupportedError


class SweBenchProAdapter(Adapter):
    name = "swe_bench_pro"

    def task_args(self, question_ids):
        raise TaskArgsUnsupportedError(
            "the SWE-Bench Pro harness (swe_bench_pro_eval.py) has no "
            "task-filter flag: it evaluates exactly the instances in the "
            "predictions JSON given via --patch_path. Restrict that file to "
            "the subset with 'zipbench-agent tasks swe_bench_pro "
            "--subset <size> --format ids'")

    def parse_results(self, path):
        path = Path(path)
        if path.is_dir():
            candidate = path / "eval_results.json"
            if not candidate.is_file():
                raise ResultParseError(f"{path} contains no eval_results.json")
            path = candidate
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            raise ResultParseError(f"cannot read {path}: {e}")

        if isinstance(data, dict) and isinstance(data.get("results"), list):
            return ParseResult(scores=self._parse_per_instance(data, path))
        if isinstance(data, dict) and data and all(
                isinstance(v, bool) for v in data.values()):
            return ParseResult(
                scores={iid: float(v) for iid, v in data.items()})
        raise ResultParseError(
            f"{path} is not a recognised SWE-Bench Pro result: expected the "
            f"harness eval_results.json (flat {{instance_id: bool}}) or a "
            f"per-instance conversion (results list)")

    @staticmethod
    def _parse_per_instance(data, path):
        scores = {}
        for row in data["results"]:
            try:
                scores[row["instance_id"]] = float(row["correct"])
            except (KeyError, TypeError) as e:
                raise ResultParseError(f"bad results row in {path}: {row!r} ({e})")
        return scores
