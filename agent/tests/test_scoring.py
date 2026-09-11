import pytest

from zipbench_agent.adapters import ParseResult
from zipbench_agent.scoring import MissingAnchorsError, score_subset
from zipbench_agent.subsets import load_subset

SUBSET = load_subset("tau2_airline", "tiny")  # 14 anchors, 50 total tasks
ANCHOR_IDS = SUBSET.question_ids


def all_ones():
    return {qid: 1.0 for qid in ANCHOR_IDS}


def test_complete_results():
    report = score_subset(SUBSET, ParseResult(scores=all_ones()))
    assert report.estimate == pytest.approx(1.0)
    assert report.n_scored == report.n_anchors == 14
    assert report.absent_anchors == report.errored_anchors == []
    assert report.scored_errored == []
    assert report.full_set_mean is None


def test_fail_on_missing():
    scores = all_ones()
    del scores["46"]
    del scores["28"]
    with pytest.raises(MissingAnchorsError) as exc:
        score_subset(SUBSET, ParseResult(scores=scores, errored={"28"}))
    msg = str(exc.value)
    assert "absent (1): 46" in msg
    assert "errored (1): 28" in msg
    assert "--missing zero" in msg
    assert exc.value.report.absent_anchors == ["46"]
    assert exc.value.report.errored_anchors == ["28"]


def test_zero_policy_keeps_weights():
    scores = all_ones()
    del scores["46"]  # weight 0.22
    report = score_subset(
        SUBSET, ParseResult(scores=scores, errored={"46"}),
        missing_policy="zero")
    assert report.estimate == pytest.approx(0.78)
    assert report.absent_anchors == []
    assert report.errored_anchors == ["46"]
    assert report.missing_policy == "zero"


def test_scored_errored_passthrough():
    # an id both scored and errored (swebench error_ids) is not missing
    scores = all_ones()
    scores["46"] = 0.0
    report = score_subset(SUBSET, ParseResult(scores=scores, errored={"46"}))
    assert report.estimate == pytest.approx(0.78)
    assert report.scored_errored == ["46"]
    assert report.errored_anchors == []


def test_bad_policy():
    with pytest.raises(ValueError):
        score_subset(SUBSET, ParseResult(scores=all_ones()),
                     missing_policy="renormalize")


def test_full_set_mean():
    scores = {str(i): 1.0 for i in range(50)}
    report = score_subset(SUBSET, ParseResult(scores=scores))
    assert report.full_set_mean == pytest.approx(1.0)


def test_full_set_mean_needs_all_anchors():
    # 50 scores, but one anchor id replaced by an extraneous id
    scores = {str(i): 1.0 for i in range(50)}
    del scores["46"]
    scores["extra-id"] = 1.0
    report = score_subset(SUBSET, ParseResult(scores=scores),
                          missing_policy="zero")
    assert report.full_set_mean is None
