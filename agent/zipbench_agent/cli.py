"""zipbench-agent CLI: subset task export and full-score reconstruction.

The upstream harnesses (swebench, harbor, tau2) are never wrapped or
re-implemented — ``tasks`` feeds them, ``score`` reads their output,
``run`` merely appends the task filter to a command you supply verbatim.
"""

import argparse
import json
import os
import sys

from . import __version__
from .adapters import ResultParseError, TaskArgsUnsupportedError, get_adapter
from .scoring import MissingAnchorsError, score_subset
from .subsets import SIZES, available_datasets, load_manifest, load_subset


def cmd_list(args):
    rows = []
    for dataset in available_datasets():
        m = load_manifest(dataset)
        rows.append((dataset, m["display_name"], str(m["n_total_questions"]),
                     str(m["subsets"]["small"]["n"]), str(m["subsets"]["tiny"]["n"]),
                     m["official_metric"]))
    headers = ("dataset", "benchmark", "full", "small", "tiny", "metric")
    widths = [max(len(r[i]) for r in rows + [headers]) for i in range(len(headers))]
    for row in [headers] + rows:
        print("  ".join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip())
    return 0


def cmd_tasks(args):
    subset = load_subset(args.dataset, args.subset)
    adapter = get_adapter(args.dataset)
    ids = subset.question_ids
    if args.format == "ids":
        for qid in ids:
            print(qid)
    elif args.format == "args":
        try:
            print(adapter.format_task_args(ids))
        except TaskArgsUnsupportedError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
    elif args.format == "json":
        print(json.dumps(ids, ensure_ascii=False, indent=2))
    elif args.format == "meta":
        try:
            print(json.dumps(adapter.task_meta(ids), ensure_ascii=False, indent=2))
        except TaskArgsUnsupportedError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
    return 0


def _fmt_ids(ids):
    return ", ".join(ids[:10]) + (" ..." if len(ids) > 10 else "")


def cmd_score(args):
    subset = load_subset(args.dataset, args.subset)
    adapter = get_adapter(args.dataset)
    try:
        parse_result = adapter.parse_results(args.results)
    except ResultParseError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    try:
        report = score_subset(subset, parse_result, missing_policy=args.missing)
    except MissingAnchorsError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        return 0
    m = subset.manifest
    print(f"dataset : {subset.dataset} ({m['display_name']})")
    print(f"subset  : {subset.size} "
          f"({report.n_anchors} anchors of {subset.n_total_questions} tasks)")
    print(f"results : {args.results} -> scored "
          f"{report.n_scored}/{report.n_anchors} anchors")
    if report.absent_anchors:
        print(f"warning : {len(report.absent_anchors)} anchors absent "
              f"(never ran): {_fmt_ids(report.absent_anchors)}; "
              f"counted as 0 by --missing zero")
    if report.errored_anchors:
        print(f"warning : {len(report.errored_anchors)} anchors errored "
              f"during evaluation: {_fmt_ids(report.errored_anchors)}; "
              f"counted as 0 by --missing zero")
    if report.scored_errored:
        print(f"note    : {len(report.scored_errored)} anchors errored during "
              f"evaluation and count 0 per the harness's official metric: "
              f"{_fmt_ids(report.scored_errored)}")
    print(f"estimated full-benchmark score ({m['official_metric']}): "
          f"{report.estimate:.4f}")
    if report.full_set_mean is not None:
        print(f"(results cover the full benchmark; actual full-set score: "
              f"{report.full_set_mean:.4f})")
    return 0


def cmd_run(args):
    subset = load_subset(args.dataset, args.subset)
    adapter = get_adapter(args.dataset)
    if not args.command:
        print("error: no command given after '--'", file=sys.stderr)
        return 2
    try:
        task_args = adapter.task_args(subset.question_ids)
    except TaskArgsUnsupportedError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    argv = list(args.command) + task_args
    print(f"+ {argv[0]} ... [{len(subset.anchors)} task filter args appended]",
          file=sys.stderr)
    try:
        os.execvp(argv[0], argv)
    except OSError as e:
        print(f"error: cannot exec {argv[0]!r}: {e}", file=sys.stderr)
        return 127


def build_parser():
    parser = argparse.ArgumentParser(
        prog="zipbench-agent",
        description="ZipBench compressed-subset evaluation for agent benchmarks.")
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list supported benchmarks and subset sizes")

    def add_common(p):
        p.add_argument("dataset", metavar="BENCH",
                       help="dataset key (see 'zipbench-agent list')")
        p.add_argument("--subset", choices=SIZES, required=True)

    p = sub.add_parser(
        "tasks", help="print the subset's task ids for the upstream CLI")
    add_common(p)
    p.add_argument("--format", choices=("ids", "args", "json", "meta"),
                   default="ids",
                   help="ids: one per line; args: shell-quoted upstream "
                        "task-filter flags; json: JSON array; meta: upstream "
                        "meta-file JSON (OSWorld --test_all_meta_path)")

    p = sub.add_parser(
        "score", help="reconstruct the full-benchmark score from native results")
    add_common(p)
    p.add_argument("--results", required=True,
                   help="native result file or directory of the upstream harness")
    p.add_argument("--missing", choices=("fail", "zero"), default="fail",
                   help="what to do with anchors that have no result: "
                        "fail (default) refuses to score and lists them; "
                        "zero counts them as 0 with their weights kept")
    p.add_argument("--json", action="store_true", help="machine-readable output")

    p = sub.add_parser(
        "run",
        help="exec an upstream command with the subset's task filter appended",
        description="Everything after '--' is executed verbatim with the "
                    "subset task-filter arguments appended, e.g.: "
                    "zipbench-agent run tau2_airline --subset tiny -- "
                    "tau2 run --domain airline --agent-llm gpt-5.4")
    add_common(p)
    return parser


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    # split off the verbatim upstream command before argparse sees it
    command = []
    if "--" in argv:
        idx = argv.index("--")
        argv, command = argv[:idx], argv[idx + 1:]
    args = build_parser().parse_args(argv)
    args.command = command
    handler = {"list": cmd_list, "tasks": cmd_tasks,
               "score": cmd_score, "run": cmd_run}[args.cmd]
    try:
        return handler(args)
    except KeyError as e:
        print(f"error: {e.args[0]}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
