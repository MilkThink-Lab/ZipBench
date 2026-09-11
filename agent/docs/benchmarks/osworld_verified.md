# OSWorld-Verified

| dataset key | full | small | tiny |
| --- | ---: | ---: | ---: |
| `osworld_verified` | 361 | 205 | 98 |

Official harness: [OSWorld](https://github.com/xlang-ai/OSWorld) @ `84aee65`
(Verified).

The subset targets OSWorld-Verified on the 361-task `test_nogdrive.json` list
(the official 369 minus 8 Google Drive tasks). Pre-Verified runs are not
score-comparable.

## Run

OSWorld selects tasks through a meta JSON (`--test_all_meta_path`), not argv —
`--format meta` emits that file:

```bash
zipbench-agent tasks osworld_verified --subset tiny --format meta > subset_meta.json

python run.py --test_all_meta_path subset_meta.json \
    --model <model> --result_dir ./results ...

zipbench-agent score osworld_verified --subset tiny --results ./results
```

## Scoring

`score` accepts the results directory (collects every
`<domain>/<uuid>/result.txt` beneath it) or the `all_result.json` written by
`show_result.py`.

Scores keep the harness's raw float reward (no binarisation, clamped to
[0, 1]); multiple runs of the same task are averaged. A `result.txt` (or
`all_result.json` entry) whose value is not a finite number — typically an
error message dumped by the harness — marks its task as errored.
