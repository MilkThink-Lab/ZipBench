# Toolathlon

| dataset key | full | small | tiny |
| --- | ---: | ---: | ---: |
| `toolathlon` | 108 | 76 | 58 |

Official harness: [Toolathlon](https://github.com/hkust-nlp/Toolathlon)
`run_parallel.py`.

The subset targets the original 108-task release (pre-Verified);
Toolathlon-Verified task sets are not score-comparable.

## Run

The harness's `--task_list` file is one task name per line — exactly the
default `--format ids` output:

```bash
zipbench-agent tasks toolathlon --subset tiny > tiny_tasks.txt

python run_parallel.py --tasks_folder finalpool --task_list tiny_tasks.txt \
    --model_short_name <model> --provider <provider> --maxstep 100 \
    --dump_path dumps/my-tiny-run

zipbench-agent score toolathlon --subset tiny --results dumps/my-tiny-run
```

## Scoring

`score` reads each `<tasks_folder>/<task_name>/eval_res.json` under the dump
directory; a task scores 1 only if its `pass` is `true` (0/1, no partial
credit), and multiple runs are averaged. A task whose `pass` is null or
absent (the evaluation produced no verdict) is reported as errored.
