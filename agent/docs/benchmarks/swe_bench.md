# SWE-bench Verified / Multilingual

| dataset key | benchmark | full | small | tiny |
| --- | --- | ---: | ---: | ---: |
| `swe_bench_verified` | SWE-bench Verified | 500 | 188 | 125 |
| `swe_bench_multilingual` | SWE-bench Multilingual | 300 | 110 | 78 |

Official harness: [swebench](https://github.com/SWE-bench/SWE-bench) >= 4.1.0.

## Run

SWE-bench is two-stage: your scaffold generates patches, the harness evaluates
them. Restrict both stages to the subset:

```bash
# 1. generate predictions only for the subset instances
zipbench-agent tasks swe_bench_verified --subset tiny > tiny_ids.txt
#    ... run your scaffold (SWE-agent, mini-swe-agent, ...) on those ids ...

# 2. evaluate with the official harness, filtered to the same ids
python -m swebench.harness.run_evaluation \
    --dataset_name SWE-bench/SWE-bench_Verified \
    --predictions_path preds.json --run_id my-tiny-run \
    $(zipbench-agent tasks swe_bench_verified --subset tiny --format args)

# 3. rescore the harness report
zipbench-agent score swe_bench_verified --subset tiny \
    --results <model>.my-tiny-run.json
```

For `swe_bench_multilingual`, use `--dataset_name SWE-bench/SWE-bench_Multilingual`
and the `swe_bench_multilingual` dataset key throughout.

## Scoring

`score` accepts the harness summary report (`resolved_ids`/`unresolved_ids`),
a `logs/run_evaluation/<run_id>/<model>/` directory of per-instance
`report.json` files, or a per-instance conversion
(`{"results": [{"instance_id", "correct"}]}`).

Unresolved, empty-patch and error instances all count 0, mirroring the
official resolved rate. `error_ids` instances are additionally flagged as
errored in the score report — they stay counted as 0 per the official
convention, the flag is diagnostic only.
