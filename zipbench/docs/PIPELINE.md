# ZipBench end-to-end compression pipeline

From raw model answer records to a compressed anchor subset:

```
raw model answer records (JSON / nested JSON+xlsx)
        │
        ├── train side: 6 real records ──ensemble voting──► 6 synthetic records ──aggregate──► <dataset>_train.pkl (12 models)
        └── test  side: real held-out records ─────────────────direct extraction────────────► <dataset>_test.pkl
        │
        ▼
IRT training ──► fixed-ratio compression (recommended ratios 0.5 / 0.75) ──► anchor subset + evaluation report
```

**train.pkl vs test.pkl**: the train side first synthesizes 6 new model
records from the 6 real records via uniform-accuracy-bin ensemble voting,
then aggregates all 12 models (6 real + 6 synthetic) into one pkl. The
test side performs no synthesis — real held-out records are extracted directly.

---

## Part 1: pkl preparation (`data_prep/`)

### 1.0 Common conventions (all modalities)

The record modalities below follow the same steps; they differ only in the
raw record layout, how correctness is determined, and which scripts are used.

Final pkl format (train and test identical):

```python
{
    "models": [model_name_0, model_name_1, ...],
    "data": {
        "<dataset>": {                     # key must equal the dataset/scenario name
            "correctness": np.ndarray      # shape (n_items, n_models), values in {0, 1}
        }
    }
}
```

**train.pkl:** validate the real records (identical question-ID sets, no
missing items) → synthesize 6 records via uniform accuracy bins (`--m 6`,
ensembles of k ≥ 2 real models) → reshape the generated JSONs into the
`model/<dataset>.json` layout with `reorganize_json_files.py` → aggregate
real + synthetic with `aggregate_results_json_multiqa.py` → validate
(`len(models) == 12`, `correctness.shape == (n_items, 12)`, values only
0/1, `data` key equals the dataset name).

**test.pkl:** validate the real held-out records → aggregate them only (no
synthetic directories) → validate as above with `len(models)` equal to the
number of real models.

**General notes:**

- Before using a new dataset, confirm the voting and aggregation scripts
  support its correctness parsing (see `DATASET_SPECS` inside the scripts);
  extend them first if not.
- Continuous-metric datasets (1.4): synthesis reads the **binarized**
  record directory; the continuous records are only used for the test split.
- Naming convention: synthetic dir `<dataset>_synthetic_m6_<tag>`; train
  pkl `<dataset>_6plus6_<tag>.pkl`; test pkl `<dataset>_test.pkl`.

### 1.1 Text benchmarks

**Record layout:** already flat, `record_root/<model>/<dataset>.json`.
Correctness is parsed from the prediction (option-letter comparison, or an
explicit correctness field for code benchmarks).

```bash
# 1) Validate real records + report accuracy

# 2) Synthesize 6 records (--strict-bins: fail loudly instead of silently
#    generating fewer records when a bin cannot be filled)
python data_prep/batch_vote_uniform_bins_results_json_multiqa.py \
  --m 6 --root '<record_root>' --dataset '<dataset>' \
  --output-dir '<synthetic_dir>' --strict-bins

# 3) Reshape into model/<dataset>.json layout
python data_prep/reorganize_json_files.py '<synthetic_dir>/generated' \
  --dataset '<dataset>' --no-confirm

# 4) Aggregate real + synthetic
python data_prep/aggregate_results_json_multiqa.py \
  --root_dir '<record_root>' '<synthetic_dir>/generated' \
  --dataset_name '<dataset>' --output_path '<train_pkl_path>'

# 5) Validate
```

**test.pkl:** no synthesis; aggregate the held-out directory directly:

```bash
python data_prep/aggregate_results_json_multiqa.py \
  --root_dir '<testset_dir>' --dataset_name '<dataset>' \
  --output_path '<...>/<dataset>_test.pkl'
```

### 1.2 Multimodal benchmarks (VLMEvalKit-style)

**Record layout:** nested directories where each model mixes judge-scoring
artifacts (`*_result.xlsx/json`). The key extra step is **normalization**:
select the scoring files by judge match and unify them into a
`model/<dataset>.json` staging directory (symlinking existing JSONs;
converting from xlsx with `--fallback-xlsx` where the JSON is missing).
After that, everything aligns with the other modalities. Requires
`openpyxl`.

> Note: the multimodal records shipped under `zipbench/data/` are a trimmed
> open-source version; fields from the raw outputs that the scripts never
> read have been removed.

```bash
# 1) Candidate discovery: confirm the judge pattern matches the expected models
python data_prep/vote_offline_results_xlsx_correctness.py \
  --root '<input_dir>' --dataset '<dataset>' \
  --judge '<judge_a>,<judge_b>' --print-candidates

# 2) Normalize the real records into a model/<dataset>.json staging dir
python data_prep/prepare_nested_json_records.py '<input_dir>' '<real_normalized_dir>' \
  --dataset '<dataset>' --judge '<judge_a>,<judge_b>' --fallback-xlsx --overwrite

# 3) Synthesize 6 records (xlsx variant of the voting script)
python data_prep/batch_vote_uniform_bins_results_xlsx_correctness.py \
  --m 6 --root '<input_dir>' --dataset '<dataset>' \
  --judge '<judge_a>,<judge_b>' --output-dir '<output_base>/synthetic_m6_<judge_tag>'

# 4) reorganize + aggregate + validate as in 1.1 steps 3)–5); the aggregate
#    --root_dir arguments are '<real_normalized_dir>' and
#    '<.../synthetic_m6_<judge_tag>/generated'
```

**test.pkl:** normalize the held-out directory (step 2), then aggregate the
normalized real records only.

### 1.3 Agentic benchmarks

**Record layout:** flat `model/<dataset>.json` with an explicit per-item
correctness field (read priority: `voted_correctness` > `any_correct` >
`is_correct`) — no prediction parsing, no normalization; the simplest of
the three.

```bash
# 1) Synthesize 6 records (same script as text)
python data_prep/batch_vote_uniform_bins_results_json_multiqa.py \
  --m 6 --root '<input_dir>' --dataset '<dataset>' --output-dir '<output_dir>'

# 2) reorganize, 3) aggregate, 4) validate — as in 1.1
```

**test.pkl:** as in 1.1.

### 1.4 Continuous-metric benchmarks

For benchmarks whose per-item metric is a float in [0, 1] (ArenaHard,
OSWorld-Verified, Terminal-Bench, the OmniDocBench sub-metrics, OCRBench_v2),
two steps precede the common pipeline:

- `convert_continuous_native_records.py` converts a harness-native result
  into the standard per-model record format.
- `binarize_continuous_records.py` turns the continuous records into 0/1
  records. Thresholds are fitted on the real training records, one per task
  group (e.g. OSWorld by domain, OCRBench_v2 by question type), choosing the
  cut whose binary accuracy best matches the continuous per-model mean.

```bash
# 0) Convert each real record
python data_prep/convert_continuous_native_records.py \
  --source '<source>' --input '<native_result>' \
  --model '<model>' --dataset '<dataset>' --output-root '<record_root>'

# 1) Binarize the real train records
python data_prep/binarize_continuous_records.py \
  --root '<record_root>' --dataset '<dataset>' \
  --output-dir '<binarized_root>' --require-count 6

# 2)–5) as in 1.1, with '<binarized_root>' as the record root
```

**test.pkl:** convert the held-out records and aggregate them without
binarization; the matrix is float:

```bash
python data_prep/aggregate_results_json_multiqa.py \
  --root_dir '<testset_dir>' --dataset_name '<dataset>' \
  --output_path '<...>/<dataset>_test.pkl' --continuous
```

**Fully continuous variant:** skip binarization, synthesize with
`generate_synthetic_continuous_results_json.py`, and aggregate with
`--continuous`; IRT training then switches to soft-label BCE automatically.

---

## Part 2: IRT training and fixed-ratio compression (`compression/`)

### Step 1 — Place the pkl files

Any location works; both scripts take explicit paths. The train and test
pkl **must use the same `data` key** (equal to the scenario name).

### Step 2 — Train IRT models

```bash
cd compression
python train_irt.py \
  --scenario <dataset> \
  --train-pkl <path>/<dataset>_train.pkl \
  --output-dir ./output/<dataset> \
  --seed 7 --run-id 1
```

Fixed training hyperparameters: dimensions D=5, epochs=2000, lr=0.1
(`--device cpu` if no GPU). Output: `./output/<dataset>/irt_model_run1/`,
which is the `--model-path` used in the next step.

For multi-task benchmarks pass every sub-task data key via
`--subscenarios`; anchors are then allocated proportionally to sub-task
size with an independent K-means per sub-task (≥ 1 anchor per sub-task).

### Step 3 — Compress at a fixed ratio

```bash
python compress_dataset.py \
  --scenario <dataset> \
  --train-data <path>/<dataset>_train.pkl \
  --model-path ./output/<dataset>/irt_model_run1/ \
  --test-data <path>/<dataset>_test.pkl \
  --ratios 0.5 0.75 \
  --seed 42 \
  --output-dir ./compressed/<dataset>
```

- `--ratios` is the fraction of items **removed**: ratio 0.5 keeps ~50% of
  the items, ratio 0.75 keeps ~25%. **Recommended: 0.5 and 0.75.** The
  anchor count is `int(round((1 - ratio) * n_items))`.
- `--seed` fixes the K-means random state (default 42) for reproducible
  anchor selection.
- Per ratio, the script writes `<dataset>_anchor_ratio_<r>.pkl` containing
  the anchor `indices` and cluster `weights` (this is the compressed
  subset), plus — when `--test-data` is given — an evaluation JSON
  `<dataset>_eval_ratio_<r>.json`.

### Step 4 — Read the evaluation report

The evaluation JSON compares, for every held-out test model, its
weighted-anchor subset score against its full-benchmark score: `mae` /
`rmse` measure the score error across test models, `spearman` is the rank
correlation between the full and subset rankings, and `per_model_errors`
gives the signed error per model. The raw full/subset accuracies and
rankings are included as well.

`eval_directly.py` offers the same evaluation as a standalone CLI over an
existing anchor file and any number of test splits
(`--test-data name=/path/to/test.pkl`, repeatable).

---

## Troubleshooting

- **`numpy._core` error when loading a pkl** — the pkl was serialized with
  numpy ≥ 2.0; re-save it in a numpy 1.x environment.
- **Spearman is None / NaN** — at extreme compression all models score the
  same and the rank correlation is undefined; use a lower ratio.
- **Fewer than 6 synthesized records** — some accuracy bins could not be
  filled from the real records. Check `<synthetic_dir>/selected_list.txt`
  and `selected_summary.jsonl` for the chosen bins; keep the smaller set
  as-is and do not top it up with extra synthesis passes.
