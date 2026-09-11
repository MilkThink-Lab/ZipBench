"""Adapter for the Toolathlon harness (hkust-nlp/Toolathlon, run_parallel.py).

The harness selects tasks through a task-list txt file (``--task_list``, one
task name per line — exactly the ``tasks toolathlon --format ids`` output), not
through argv, so ``task_args`` raises.

Recognised result inputs: the run's dump directory. Each evaluated task leaves
``{dump_path}/{tasks_folder}/{task_name}/eval_res.json`` whose ``"pass"`` key
is the boolean verdict (0/1, no partial credit). The task's identity is the
directory name — ``eval_res.json`` itself carries no task-name key, which is
also why the aggregated ``eval_res_all.jsonl`` (a plain concatenation of those
files) cannot be mapped back to tasks and is *not* accepted here. Like the
official stats script, a ``SingleUserTurn-`` directory-name prefix is
stripped. A task whose ``pass`` is null/absent (the evaluation never produced
a verdict) is reported as errored; several results for the same task name are
averaged.
"""

import json
from pathlib import Path

from .base import Adapter, ParseResult, ResultParseError, TaskArgsUnsupportedError


class ToolathlonAdapter(Adapter):
    name = "toolathlon"

    def task_args(self, question_ids):
        raise TaskArgsUnsupportedError(
            "the Toolathlon harness has no task-filter flag: run_parallel.py "
            "takes a task-list file via --task_list. Generate it with "
            "'zipbench-agent tasks toolathlon --subset <size> "
            "--format ids > tasks.txt' and pass '--task_list tasks.txt'")

    def parse_results(self, path):
        path = Path(path)
        if not path.is_dir():
            raise ResultParseError(
                f"{path} is not a directory: pass the Toolathlon dump "
                f"directory (its <tasks_folder>/<task_name>/eval_res.json "
                f"files carry no task-name key, so single files cannot be "
                f"attributed to a task)")
        successes = {}  # task_name -> [0.0/1.0, ...]
        errored = set()
        for eval_res in sorted(path.rglob("eval_res.json")):
            try:
                data = json.loads(eval_res.read_text())
            except (OSError, json.JSONDecodeError) as e:
                raise ResultParseError(f"cannot read {eval_res}: {e}")
            if not isinstance(data, dict):
                raise ResultParseError(
                    f"{eval_res} is not a recognised Toolathlon eval_res.json "
                    f"(expected an object with a 'pass' key)")
            task = eval_res.parent.name
            if task.startswith("SingleUserTurn-"):
                task = task[len("SingleUserTurn-"):]
            verdict = data.get("pass")
            if not isinstance(verdict, bool):
                errored.add(task)  # null/absent: evaluation produced no verdict
                continue
            successes.setdefault(task, []).append(1.0 if verdict else 0.0)
        if not successes and not errored:
            raise ResultParseError(
                f"{path} contains no <task_name>/eval_res.json files with a "
                f"boolean 'pass' verdict")
        scores = {task: sum(s) / len(s) for task, s in successes.items()}
        return ParseResult(scores=scores, errored=errored - scores.keys())
