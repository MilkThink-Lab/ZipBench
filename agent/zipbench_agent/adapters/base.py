"""Adapter interface: one class per upstream harness.

An adapter owns exactly two responsibilities — turning a subset's task ids
into the upstream CLI's task-filter arguments, and turning the upstream's
native result files back into per-task scores. It never runs the harness.
"""

import shlex
from dataclasses import dataclass, field


class ResultParseError(Exception):
    pass


class TaskArgsUnsupportedError(Exception):
    """The upstream harness has no inline task-filter flag.

    Raised by adapters whose harness selects tasks through a file (a
    predictions JSON, a meta JSON, a task-list txt) rather than argv; the
    message explains which ``tasks`` output feeds that file.
    """


@dataclass
class ParseResult:
    """What ``parse_results`` returns.

    ``scores`` maps question_id -> score in [0, 1]. ``errored`` holds task
    ids that left a trace in the results but produced no valid score (the
    evaluation crashed or never emitted a verdict). A task with at least one
    valid score is scored, not errored — adapters keep the two disjoint
    (multi-trial runs drop a task from ``errored`` once any trial scored).

    Exception: an adapter may put an id in both sets when the harness's
    official metric already counts an evaluation error as 0 (currently only
    swebench's ``error_ids``). Such ids are *not* missing — scoring uses the
    0 recorded in ``scores`` and only surfaces the overlap as a diagnostic.
    Missing is always judged by ``scores`` alone; ``errored`` merely splits
    the missing ids into "errored" vs "absent" and flags scored errors.
    """
    scores: dict = field(default_factory=dict)
    errored: set = field(default_factory=set)


class Adapter:
    #: dataset key, e.g. "terminal_bench"
    name = None

    def task_args(self, question_ids):
        """Return the upstream task-filter arguments as an argv list."""
        raise NotImplementedError

    def task_meta(self, question_ids):
        """Return the subset as the upstream's meta-file JSON object.

        Only meaningful for harnesses whose task filter is a meta file
        (currently OSWorld's ``--test_all_meta_path``).
        """
        raise TaskArgsUnsupportedError(
            f"{self.name} has no meta-file task format; use "
            f"'zipbench-agent tasks {self.name} --format args' (inline "
            f"task-filter flags) or '--format ids' instead")

    def parse_results(self, path):
        """Parse a native result file/dir into a ParseResult (see above)."""
        raise NotImplementedError

    def format_task_args(self, question_ids):
        """The task-filter arguments as a single shell-quoted string."""
        return " ".join(shlex.quote(a) for a in self.task_args(question_ids))
