"""Adapter for τ³-bench (telecom / retail / airline domains; the upstream
package and CLI are still named ``tau2``, and tau2 v1.0.0 is the τ³ release).

Recognised result inputs: a run directory (``data/simulations/<run>/``) or its
``results.json``, in the monolithic format with a ``simulations`` list. A task
succeeds when ``reward_info.reward == 1.0`` (tau2's own criterion); with
several trials per task the per-trial successes are averaged. Simulations
without a reward (infrastructure errors) produce no score, matching tau2's
metric computation; a task whose every simulation died that way is reported
as errored.
"""

import json
from pathlib import Path

from .base import Adapter, ParseResult, ResultParseError

SUCCESS_EPS = 1e-6


class Tau2Adapter(Adapter):

    def __init__(self, name, domain):
        self.name = name
        self.domain = domain

    def task_args(self, question_ids):
        return ["--task-ids"] + list(question_ids)

    def parse_results(self, path):
        path = Path(path)
        if path.is_dir():
            candidate = path / "results.json"
            if not candidate.is_file():
                raise ResultParseError(f"{path} contains no results.json")
            path = candidate
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            raise ResultParseError(f"cannot read {path}: {e}")
        simulations = data.get("simulations")
        if not isinstance(simulations, list):
            raise ResultParseError(
                f"{path} is not a recognised tau2 results.json "
                f"(no 'simulations' list)")

        successes = {}  # task_id -> [0/1 per trial]
        errored = set()
        for sim in simulations:
            task_id = sim.get("task_id")
            if task_id is None:
                continue
            reward_info = sim.get("reward_info") or {}
            reward = reward_info.get("reward")
            if reward is None:
                errored.add(str(task_id))  # infrastructure error: no score
                continue
            success = 1.0 if float(reward) >= 1.0 - SUCCESS_EPS else 0.0
            successes.setdefault(str(task_id), []).append(success)
        if not successes and not errored:
            raise ResultParseError(f"{path} contains no scored simulations")
        scores = {task: sum(s) / len(s) for task, s in successes.items()}
        return ParseResult(scores=scores, errored=errored - scores.keys())
