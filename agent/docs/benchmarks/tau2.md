# τ³-bench (telecom / retail / airline)

| dataset key | domain | full | small | tiny |
| --- | --- | ---: | ---: | ---: |
| `tau2_telecom` | telecom (task split `base`) | 114 | 52 | 22 |
| `tau2_retail` | retail | 114 | 79 | 72 |
| `tau2_airline` | airline | 50 | 23 | 14 |

Official harness: [tau2](https://github.com/sierra-research/tau2-bench) >= 1.0.0.

Naming note: these subsets target τ³-bench. The upstream `tau2-bench` repo's
v1.0.0 release *is* τ³-bench (the pip package and CLI are still named `tau2`).
The τ³ task sets include the 75+ task fixes and are not score-comparable with
pre-τ³ (<= 0.2.x) runs.

## Run

```bash
tau2 run --domain airline --agent-llm <model> --user-llm <model> \
    $(zipbench-agent tasks tau2_airline --subset tiny --format args)

zipbench-agent score tau2_airline --subset tiny \
    --results data/simulations/<run-dir>
```

`--format args` emits `--task-ids ...` (telecom ids are shell-quoted — always
use the emitted quoting). Telecom uses the CLI-default task split `base`
(114 tasks).

## Scoring

A task counts as solved when `reward == 1.0` (τ³'s own criterion); multiple
trials are averaged, and simulations that died on infrastructure errors
produce no score, matching `tau2`'s metric computation. A task whose every
simulation lacks a reward is reported as errored.
