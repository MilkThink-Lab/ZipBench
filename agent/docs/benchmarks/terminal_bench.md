# Terminal-Bench 2.0

| dataset key | full | small | tiny |
| --- | ---: | ---: | ---: |
| `terminal_bench` | 89 | 72 | 50 |

Official harness: [harbor](https://github.com/laude-institute/harbor) >= 0.4.0,
dataset `terminal-bench@2.0`.

## Run

```bash
harbor run --dataset terminal-bench@2.0 --agent <agent> --model <model> \
    --job-name my-tiny-run --jobs-dir jobs \
    $(zipbench-agent tasks terminal_bench --subset tiny --format args)

zipbench-agent score terminal_bench --subset tiny --results jobs/my-tiny-run
```

`--format args` emits repeatable `-i <task-name>` filters.

## Scoring

`score` accepts a job directory, a job `result.json`, a directory of trial
dirs, or a single trial `result.json`.

Per the ZipBench convention the trial reward is binarised at >= 0.5 (harbor
rewards for `terminal-bench@2.0` are effectively 0/1 already); multiple
attempts per task are averaged after binarisation.

A trial `result.json` with a `task_name` but no verifier reward (the trial
crashed before verification) marks its task as errored. The job-level
`result.json` aggregate only lists rewarded trials, so for a job directory
the trial subdirectories are still scanned for crashed trials; a bare job
`result.json` file carries no such trace.
