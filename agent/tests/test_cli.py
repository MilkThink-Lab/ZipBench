import json

from zipbench_agent.cli import main
from zipbench_agent.subsets import load_subset

ANCHOR_IDS = load_subset("tau2_airline", "tiny").question_ids


def write_results(tmp_path, sims):
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"simulations": [
        {"task_id": tid, "reward_info": {"reward": r}} for tid, r in sims]}))
    return path


def test_score_complete(tmp_path, capsys):
    path = write_results(tmp_path, [(qid, 1.0) for qid in ANCHOR_IDS])
    assert main(["score", "tau2_airline", "--subset", "tiny",
                 "--results", str(path)]) == 0
    out = capsys.readouterr().out
    assert "scored 14/14 anchors" in out
    assert "1.0000" in out


def test_score_missing_fails(tmp_path, capsys):
    sims = [(qid, 1.0) for qid in ANCHOR_IDS if qid not in ("46", "28")]
    sims.append(("28", None))  # errored trace, "46" absent
    path = write_results(tmp_path, sims)
    assert main(["score", "tau2_airline", "--subset", "tiny",
                 "--results", str(path)]) == 1
    err = capsys.readouterr().err
    assert "absent (1): 46" in err
    assert "errored (1): 28" in err
    assert "--missing zero" in err


def test_score_missing_zero(tmp_path, capsys):
    sims = [(qid, 1.0) for qid in ANCHOR_IDS if qid not in ("46", "28")]
    sims.append(("28", None))
    path = write_results(tmp_path, sims)
    assert main(["score", "tau2_airline", "--subset", "tiny",
                 "--results", str(path), "--missing", "zero"]) == 0
    out = capsys.readouterr().out
    assert "1 anchors absent (never ran): 46" in out
    assert "1 anchors errored during evaluation: 28" in out
    assert "counted as 0 by --missing zero" in out
    # weights of "46" (0.22) and "28" (0.1) drop to 0, no renormalisation
    assert "0.6800" in out


def test_score_json(tmp_path, capsys):
    sims = [(qid, 1.0) for qid in ANCHOR_IDS if qid != "46"]
    path = write_results(tmp_path, sims)
    assert main(["score", "tau2_airline", "--subset", "tiny",
                 "--results", str(path), "--missing", "zero", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["absent_anchors"] == ["46"]
    assert report["errored_anchors"] == []
    assert report["scored_errored"] == []
    assert report["missing_policy"] == "zero"
    assert abs(report["estimated_full_score"] - 0.78) < 1e-9
