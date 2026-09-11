"""Weighted reconstruction of full-benchmark scores from subset results."""

from dataclasses import dataclass, field


def _fmt_ids(ids):
    return ", ".join(ids[:10]) + (" ..." if len(ids) > 10 else "")


class MissingAnchorsError(Exception):
    def __init__(self, report):
        self.report = report
        absent, errored = report.absent_anchors, report.errored_anchors
        n_missing = len(absent) + len(errored)
        lines = [f"{n_missing}/{report.n_anchors} anchor tasks have no result."]
        if absent:
            lines.append(
                f"  absent ({len(absent)}): {_fmt_ids(absent)}\n"
                f"    These tasks left no trace in the results — they likely "
                f"never ran. Run them and rescore.")
        if errored:
            lines.append(
                f"  errored ({len(errored)}): {_fmt_ids(errored)}\n"
                f"    The evaluation errored on these tasks without producing "
                f"a score. Check the evaluation environment and re-evaluate.")
        lines.append(
            "If these tasks genuinely failed, pass --missing zero to force "
            "scoring with them counted as 0.")
        super().__init__("\n".join(lines))


@dataclass
class ScoreReport:
    dataset: str
    size: str
    n_anchors: int
    n_scored: int
    estimate: float
    absent_anchors: list = field(default_factory=list)
    errored_anchors: list = field(default_factory=list)
    scored_errored: list = field(default_factory=list)
    missing_policy: str = "fail"
    full_set_mean: float = None  # set when the results happen to cover the full benchmark

    def to_dict(self):
        return {
            "dataset": self.dataset,
            "subset": self.size,
            "n_anchors": self.n_anchors,
            "n_scored": self.n_scored,
            "estimated_full_score": self.estimate,
            "absent_anchors": self.absent_anchors,
            "errored_anchors": self.errored_anchors,
            "scored_errored": self.scored_errored,
            "missing_policy": self.missing_policy,
            "full_set_mean": self.full_set_mean,
        }


def score_subset(subset, parse_result, missing_policy="fail"):
    """Estimate the full-benchmark score from per-task results.

    ``parse_result`` is an adapters.ParseResult (extra ids are ignored). An
    anchor is missing when its id is absent from ``parse_result.scores``;
    ``parse_result.errored`` splits the missing anchors into errored (the
    evaluation left a trace but no score) and absent (no trace at all).
    ``missing_policy`` is "fail" (raise MissingAnchorsError on any missing
    anchor) or "zero" (count missing anchors as 0 with their weights kept —
    never renormalised). Ids both scored and errored (swebench error_ids)
    are not missing; they are reported in ``scored_errored``.
    """
    if missing_policy not in ("fail", "zero"):
        raise ValueError(
            f"unknown missing policy {missing_policy!r}; expected 'fail' or 'zero'")
    scores, errored = parse_result.scores, parse_result.errored
    present = [a for a in subset.anchors if a.question_id in scores]
    missing = [a.question_id for a in subset.anchors if a.question_id not in scores]

    report = ScoreReport(
        dataset=subset.dataset, size=subset.size,
        n_anchors=len(subset.anchors), n_scored=len(present),
        estimate=None, missing_policy=missing_policy,
        absent_anchors=[q for q in missing if q not in errored],
        errored_anchors=[q for q in missing if q in errored],
        scored_errored=[a.question_id for a in present
                        if a.question_id in errored])
    if missing and missing_policy == "fail":
        raise MissingAnchorsError(report)

    # missing anchors contribute 0 with their weight kept; the denominator
    # is the full subset's weight sum (== 1 up to float error) either way
    weight_sum = sum(a.weight for a in subset.anchors)
    report.estimate = sum(
        a.weight * scores[a.question_id] for a in present) / weight_sum
    if (len(scores) == subset.n_total_questions
            and all(a.question_id in scores for a in subset.anchors)):
        report.full_set_mean = sum(scores.values()) / len(scores)
    return report
