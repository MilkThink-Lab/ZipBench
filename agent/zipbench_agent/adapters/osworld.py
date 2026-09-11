"""Adapter for the OSWorld harness (xlang-ai/OSWorld, run.py + show_result.py).

OSWorld selects tasks through a meta JSON (``--test_all_meta_path``, shape
``{domain: [example_id, ...]}``), not through argv — so ``task_args`` raises
and ``--format meta`` emits that JSON (``task_meta``). The packaged
``task_domains.json`` maps each of the 361 Verified example uuids to its
domain.

Recognised result inputs:

1. An ``all_result.json`` written by ``show_result.py`` — shape
   ``{domain: {example_id: score}}`` (or already-flat ``{example_id: score}``).
   Note the harness writes it with ``str(dict)``, i.e. a Python literal with
   single quotes, so parsing falls back to ``ast.literal_eval``.
2. A results directory — either the directory holding ``all_result.json``
   directly, or any ancestor of the per-task dirs
   ``.../<model>/<domain>/<example_id>/result.txt``; every ``result.txt``
   whose parent directory name is an example uuid is collected. Files are
   parsed as ``float(text)`` with an ``ast.literal_eval`` fallback (mirroring
   show_result.py's ``float``/``eval`` fallback).

A result.txt (or all_result.json entry) whose value cannot be parsed as a
finite number — typically an error message dumped by the harness — marks its
example as errored. Scores keep the harness's raw float reward (clamped to
[0, 1]) — no binarisation; several results for the same uuid (multiple runs)
are averaged.
"""

import ast
import json
import math
import re
from importlib import resources
from pathlib import Path

from .base import Adapter, ParseResult, ResultParseError, TaskArgsUnsupportedError

UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)


class OSWorldAdapter(Adapter):
    name = "osworld_verified"

    def __init__(self):
        self._domains = None

    def _domain_map(self):
        if self._domains is None:
            path = (resources.files("zipbench_agent") / "subsets_data"
                    / self.name / "task_domains.json")
            if not path.is_file():
                raise RuntimeError(
                    f"packaging error: {self.name}/task_domains.json is "
                    f"missing from subsets_data")
            self._domains = json.loads(path.read_text())
        return self._domains

    def task_args(self, question_ids):
        raise TaskArgsUnsupportedError(
            "the OSWorld harness has no task-filter flag: run.py takes a "
            "meta JSON via --test_all_meta_path. Generate it with "
            "'zipbench-agent tasks osworld_verified --subset <size> "
            "--format meta > subset_meta.json'")

    def task_meta(self, question_ids):
        domains = self._domain_map()
        meta = {}
        for qid in question_ids:
            try:
                domain = domains[qid]
            except KeyError:
                raise RuntimeError(
                    f"packaging error: task {qid!r} has no domain in "
                    f"{self.name}/task_domains.json")
            meta.setdefault(domain, []).append(qid)
        return {domain: meta[domain] for domain in sorted(meta)}

    def parse_results(self, path):
        path = Path(path)
        rewards = {}  # example_id -> [score, ...]
        errored = set()
        if path.is_dir():
            all_result = path / "all_result.json"
            if all_result.is_file():
                self._collect_all_result(all_result, rewards, errored)
            else:
                for result_txt in sorted(path.rglob("result.txt")):
                    example_id = result_txt.parent.name
                    if not UUID_RE.fullmatch(example_id):
                        continue
                    try:
                        text = result_txt.read_text()
                    except OSError as e:
                        raise ResultParseError(f"cannot read {result_txt}: {e}")
                    score = self._parse_scalar(text.strip())
                    if score is None:
                        errored.add(example_id)  # error dump, not a reward
                    else:
                        rewards.setdefault(example_id, []).append(score)
        else:
            self._collect_all_result(path, rewards, errored)
        if not rewards and not errored:
            raise ResultParseError(
                f"{path} is not a recognised OSWorld result: expected an "
                f"all_result.json ({{domain: {{example_id: score}}}}) or a "
                f"results directory containing <domain>/<example_id>/result.txt "
                f"files")
        scores = {
            example_id: sum(rs) / len(rs)
            for example_id, rs in rewards.items()}
        return ParseResult(scores=scores, errored=errored - scores.keys())

    def _collect_all_result(self, path, rewards, errored):
        try:
            text = path.read_text()
        except OSError as e:
            raise ResultParseError(f"cannot read {path}: {e}")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # show_result.py writes all_result.json with str(dict): a Python
            # literal (single quotes), not JSON
            try:
                data = ast.literal_eval(text)
            except (ValueError, SyntaxError) as e:
                raise ResultParseError(f"cannot parse {path}: {e}")
        if not isinstance(data, dict) or not data:
            raise ResultParseError(
                f"{path} is not a recognised OSWorld all_result.json: "
                f"expected {{domain: {{example_id: score}}}} or "
                f"{{example_id: score}}")
        if all(isinstance(v, dict) for v in data.values()):
            items = [(eid, s) for domain in data.values()
                     for eid, s in domain.items()]
        else:
            items = list(data.items())
        for example_id, score in items:
            score = self._parse_scalar(score)
            if score is None:
                errored.add(example_id)
            else:
                rewards.setdefault(example_id, []).append(score)

    @staticmethod
    def _parse_scalar(value):
        """Parse one reward, clamped to [0, 1]; None if unparsable."""
        try:
            score = float(value)
        except (TypeError, ValueError):
            try:
                score = float(ast.literal_eval(value))
            except (TypeError, ValueError, SyntaxError):
                return None
        if not math.isfinite(score):
            return None
        return min(1.0, max(0.0, score))
