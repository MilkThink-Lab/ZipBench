import json

import pytest

from zipbench_agent.adapters import ResultParseError, get_adapter


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


# --- terminal_bench -------------------------------------------------------

def trial(tmp_path, name, task, reward):
    write_json(tmp_path / name / "result.json", {
        "task_name": task,
        "verifier_result": {"rewards": {"reward": reward}}})


def test_terminal_bench_trials(tmp_path):
    trial(tmp_path, "t1", "task-a", 1.0)
    trial(tmp_path, "t2", "task-b", 0.0)
    res = get_adapter("terminal_bench").parse_results(tmp_path)
    assert res.scores == {"task-a": 1.0, "task-b": 0.0}
    assert res.errored == set()


def test_terminal_bench_errored_trial(tmp_path):
    trial(tmp_path, "t1", "task-a", 1.0)
    trial(tmp_path, "t2", "task-b", None)
    res = get_adapter("terminal_bench").parse_results(tmp_path)
    assert res.scores == {"task-a": 1.0}
    assert res.errored == {"task-b"}


def test_terminal_bench_partial_trials_score(tmp_path):
    # one errored + one scored trial for the same task: scored, not errored
    trial(tmp_path, "t1", "task-a", None)
    trial(tmp_path, "t2", "task-a", 1.0)
    res = get_adapter("terminal_bench").parse_results(tmp_path)
    assert res.scores == {"task-a": 1.0}
    assert res.errored == set()


def test_terminal_bench_job_result(tmp_path):
    write_json(tmp_path / "result.json", {
        "stats": {"evals": {"e": {"reward_stats": {"reward": {
            "1.0": ["task-a__abc123"], "0.0": ["task-b__def456"]}}}}}})
    res = get_adapter("terminal_bench").parse_results(tmp_path)
    assert res.scores == {"task-a": 1.0, "task-b": 0.0}
    assert res.errored == set()


def test_terminal_bench_job_dir_with_crashed_trial(tmp_path):
    # job buckets only list rewarded trials; the crashed trial's own
    # result.json in the same job directory must still surface as errored,
    # and the rewarded trials must not be double counted
    write_json(tmp_path / "result.json", {
        "stats": {"evals": {"e": {"reward_stats": {"reward": {
            "1.0": ["task-a__abc123"]}}}}}})
    trial(tmp_path, "task-a__abc123", "task-a", 1.0)
    trial(tmp_path, "task-c__ghi789", "task-c", None)
    res = get_adapter("terminal_bench").parse_results(tmp_path)
    assert res.scores == {"task-a": 1.0}
    assert res.errored == {"task-c"}


# --- tau2 -----------------------------------------------------------------

def tau2_results(tmp_path, sims):
    path = tmp_path / "results.json"
    write_json(path, {"simulations": [
        {"task_id": tid, "reward_info": {"reward": r}} for tid, r in sims]})
    return path


def test_tau2_scores(tmp_path):
    path = tau2_results(tmp_path, [("1", 1.0), ("1", 0.0), ("2", 0.0)])
    res = get_adapter("tau2_airline").parse_results(path)
    assert res.scores == {"1": 0.5, "2": 0.0}
    assert res.errored == set()


def test_tau2_errored(tmp_path):
    path = tau2_results(tmp_path, [("1", 1.0), ("2", None), ("3", None), ("3", 0.0)])
    res = get_adapter("tau2_airline").parse_results(path)
    assert res.scores == {"1": 1.0, "3": 0.0}
    assert res.errored == {"2"}  # "3" has a scored trial, so not errored


# --- toolathlon -----------------------------------------------------------

def test_toolathlon(tmp_path):
    write_json(tmp_path / "pool" / "SingleUserTurn-foo" / "eval_res.json",
               {"pass": True})
    write_json(tmp_path / "pool" / "bar" / "eval_res.json", {"pass": False})
    write_json(tmp_path / "pool" / "baz" / "eval_res.json", {"pass": None})
    res = get_adapter("toolathlon").parse_results(tmp_path)
    assert res.scores == {"foo": 1.0, "bar": 0.0}
    assert res.errored == {"baz"}


# --- osworld --------------------------------------------------------------

UUID1 = "00000000-0000-0000-0000-000000000001"
UUID2 = "00000000-0000-0000-0000-000000000002"


def test_osworld_result_txt(tmp_path):
    d = tmp_path / "model" / "chrome"
    (d / UUID1).mkdir(parents=True)
    (d / UUID1 / "result.txt").write_text("1.0")
    (d / UUID2).mkdir(parents=True)
    (d / UUID2 / "result.txt").write_text("Traceback: something broke")
    res = get_adapter("osworld_verified").parse_results(tmp_path)
    assert res.scores == {UUID1: 1.0}
    assert res.errored == {UUID2}


def test_osworld_all_result(tmp_path):
    write_json(tmp_path / "all_result.json",
               {"chrome": {UUID1: 0.5, UUID2: "unparsable"}})
    res = get_adapter("osworld_verified").parse_results(tmp_path)
    assert res.scores == {UUID1: 0.5}
    assert res.errored == {UUID2}


# --- swe_bench ------------------------------------------------------------

def test_swe_bench_summary_report(tmp_path):
    path = tmp_path / "model.run.json"
    write_json(path, {"resolved_ids": ["a"], "unresolved_ids": ["b"],
                      "error_ids": ["c"], "empty_patch_ids": ["d"]})
    res = get_adapter("swe_bench_verified").parse_results(path)
    assert res.scores == {"a": 1.0, "b": 0.0, "c": 0.0, "d": 0.0}
    assert res.errored == {"c"}  # counted 0 *and* flagged errored


def test_swe_bench_log_dir(tmp_path):
    write_json(tmp_path / "a" / "report.json", {"a": {"resolved": True}})
    write_json(tmp_path / "b" / "report.json", {"b": {"resolved": False}})
    res = get_adapter("swe_bench_verified").parse_results(tmp_path)
    assert res.scores == {"a": 1.0, "b": 0.0}
    assert res.errored == set()


def test_swe_bench_log_dir_bad_json(tmp_path):
    write_json(tmp_path / "a" / "report.json", {"a": {"resolved": True}})
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "report.json").write_text("{not json")
    with pytest.raises(ResultParseError):
        get_adapter("swe_bench_verified").parse_results(tmp_path)


# --- swe_bench_pro --------------------------------------------------------

def test_swe_bench_pro(tmp_path):
    write_json(tmp_path / "eval_results.json", {"x": True, "y": False})
    res = get_adapter("swe_bench_pro").parse_results(tmp_path)
    assert res.scores == {"x": 1.0, "y": 0.0}
    assert res.errored == set()


def test_swe_bench_pro_bad_input(tmp_path):
    path = tmp_path / "eval_results.json"
    write_json(path, {"x": 0.5})
    with pytest.raises(ResultParseError):
        get_adapter("swe_bench_pro").parse_results(path)
