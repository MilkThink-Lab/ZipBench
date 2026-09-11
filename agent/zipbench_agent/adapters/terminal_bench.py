"""Adapter for the harbor harness (terminal-bench 2.x).

Recognised result inputs:

1. A job directory (``jobs/<job_name>/``) or its ``result.json`` — reads the
   ``stats.evals[*].reward_stats.reward`` buckets, which map each reward value
   to the trial names that received it.
2. A directory of trial dirs, or a single trial ``result.json`` — reads
   ``verifier_result.rewards.reward``.

Per the ZipBench convention, rewards are binarised at >= 0.5 (harbor rewards
for terminal-bench@2.0 are effectively 0/1 already); with several attempts
per task the binarised attempt scores are averaged.

A trial result.json with a ``task_name`` but no verifier reward (the trial
crashed before verification) marks its task as errored. The job-level
``reward_stats.reward`` buckets only list trials that received a reward, so
for a job directory the trial subdirectories are still scanned to pick up
crashed trials; only a bare job result.json file carries no errored
information.
"""

import json
from pathlib import Path

from .base import Adapter, ParseResult, ResultParseError

REWARD_THRESHOLD = 0.5


def _task_name_of_trial(trial_name):
    # trial names look like "<task-name>__<shortuuid>"
    return trial_name.rsplit("__", 1)[0]


class TerminalBenchAdapter(Adapter):
    name = "terminal_bench"

    def task_args(self, question_ids):
        args = []
        for qid in question_ids:
            args += ["-i", qid]
        return args

    def parse_results(self, path):
        path = Path(path)
        rewards = {}  # task -> [reward, ...]
        errored = set()
        if path.is_dir():
            job_result = path / "result.json"
            if job_result.is_file() and self._collect_job(job_result, rewards):
                # the job buckets only list rewarded trials; crashed trials
                # still leave a trial result.json — harvest errored only
                # (rewards go to a scratch dict to avoid double counting)
                for trial_result in sorted(path.glob("*/result.json")):
                    self._collect_trial(trial_result, {}, errored)
            else:
                for trial_result in sorted(path.glob("*/result.json")):
                    self._collect_trial(trial_result, rewards, errored)
        elif path.name.endswith(".json"):
            if not self._collect_job(path, rewards):
                self._collect_trial(path, rewards, errored)
        if not rewards and not errored:
            raise ResultParseError(
                f"{path} is not a recognised harbor result: expected a job "
                f"directory, a job/trial result.json, or a directory of trial dirs")
        scores = {
            task: sum(1.0 if r >= REWARD_THRESHOLD else 0.0 for r in rs) / len(rs)
            for task, rs in rewards.items()}
        return ParseResult(scores=scores, errored=errored - scores.keys())

    @staticmethod
    def _collect_job(path, rewards):
        """Harvest a job-level result.json; returns False if it isn't one."""
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            raise ResultParseError(f"cannot read {path}: {e}")
        evals = (data.get("stats") or {}).get("evals")
        if not isinstance(evals, dict):
            return False
        for ev in evals.values():
            buckets = ((ev.get("reward_stats") or {}).get("reward") or {})
            for reward_value, trial_names in buckets.items():
                for trial_name in trial_names:
                    task = _task_name_of_trial(trial_name)
                    rewards.setdefault(task, []).append(float(reward_value))
        return True

    @staticmethod
    def _collect_trial(path, rewards, errored):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            raise ResultParseError(f"cannot read {path}: {e}")
        task = data.get("task_name")
        if task is None:
            return  # not a trial result at all
        verifier = data.get("verifier_result") or {}
        reward = (verifier.get("rewards") or {}).get("reward")
        if reward is None:
            errored.add(task)  # trial crashed before producing a reward
            return
        rewards.setdefault(task, []).append(float(reward))
