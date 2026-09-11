# SWE-Bench Pro (public set)

| dataset key | full | small | tiny |
| --- | ---: | ---: | ---: |
| `swe_bench_pro` | 731 | 366 | 160 |

Official harness: [swe_bench_pro_eval.py](https://github.com/scaleapi/SWE-bench_Pro-os)
@ `ca10a60`.

## Run

Like SWE-bench, SWE-Bench Pro is two-stage — but its harness has no
task-filter flag: it evaluates exactly the instances listed in the predictions
JSON (`--patch_path`). The evaluated set *is* the predictions set, so restrict
the predictions to the subset ids:

```bash
# 1. generate patches only for the subset instances
zipbench-agent tasks swe_bench_pro --subset tiny > tiny_ids.txt
#    ... run your scaffold on those ids -> preds.json
#        ([{"instance_id", "patch", "prefix"}, ...]) ...

# 2. evaluate — the harness scores exactly what preds.json contains
python swe_bench_pro_eval.py --raw_sample_path <samples.csv> \
    --patch_path preds.json --output_dir out/ ...

# 3. rescore the harness output
zipbench-agent score swe_bench_pro --subset tiny --results out/eval_results.json
```

## Scoring

`score` accepts the harness's `eval_results.json` (flat
`{instance_id: true/false}`, or the directory containing it) or a
per-instance conversion (`{"results": [{"instance_id", "correct"}]}`).

An instance scores 1 only if the harness marked it resolved (`true`).

Neither format leaves any trace of evaluation errors (the flat bool mapping
just omits the instance), so missing instances all report as absent — this
harness's results cannot be split into absent vs errored.
