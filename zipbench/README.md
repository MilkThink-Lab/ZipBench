<div align="center">
  <a href="../README.md"><img src="../assets/logo.png" width="360" alt="ZipBench"></a>
  <h2>🧰 Build a Compact Benchmark with ZipBench</h2>
  <p>Reproduce the method or compress a new benchmark of your own.</p>
  <p>
    <img src="https://img.shields.io/badge/Examples-Text%20%7C%20Multimodal%20%7C%20Agent-b8860b" alt="Text multimodal and agent examples">
    <img src="https://img.shields.io/badge/Method-2PL%20IRT%20%2B%20K--means-6f42c1" alt="2PL IRT and K-means">
    <img src="https://img.shields.io/badge/Python-3.8--3.11-2f6f9f" alt="Python 3.8 to 3.11">
  </p>
  <p>
    <a href="#fastest-reproduction">Fastest Reproduction</a> ·
    <a href="#included-examples">Examples</a> ·
    <a href="#compress-a-new-benchmark">New Benchmark</a> ·
    <a href="docs/PIPELINE.md">Detailed Pipeline</a> ·
    <a href="../agent/README.md">Agent Evaluation</a> ·
    <a href="../README.md">Back to ZipBench</a>
  </p>
</div>

Use this folder when you want to **reproduce how ZipBench builds `small` and `tiny` benchmarks** or **compress a new benchmark of your own**.

> [!NOTE]
> You do not need this folder for normal model evaluation. To use the released compact benchmarks, go to [`../opencompass/`](../opencompass/README.md) for text models, [`../VLMEvalKit/`](../VLMEvalKit/README.md) for vision-language models, or [`../agent/`](../agent/README.md) for agents.

In plain terms, the construction pipeline takes item-level results from a few models, learns which benchmark samples provide similar information, and keeps one weighted representative from each group.

<div align="center">
  <b>Item-level model results</b> &nbsp;→&nbsp; <b>learn sample behavior</b> &nbsp;→&nbsp; <b>select representative samples</b> &nbsp;→&nbsp; <b>validate against the full benchmark</b>
</div>

<a id="fastest-reproduction"></a>

## 🚀 Fastest Reproduction

Prepared LiveCodeBench train and test files are included. The commands below reproduce IRT training, compact-benchmark construction, and held-out validation without preparing raw model records.

### 1. Install the construction environment

```bash
conda create -n zipbench-build python=3.11 -y
conda activate zipbench-build
cd zipbench
pip install -r requirements.txt
pip install -e ./py-irt
```

> [!IMPORTANT]
> Install the bundled `py-irt` rather than the PyPI release. Keep Python below 3.12 and NumPy below 2.0, as specified by the repository requirements.

### 2. Train the sample representations

```bash
python compression/train_irt.py \
  --scenario LiveCodeBench \
  --train-pkl data/LiveCodeBench/LiveCodeBench_train.pkl \
  --output-dir runs/LiveCodeBench \
  --device cuda \
  --seed 7 \
  --run-id 1
```

Use `--device cpu` when CUDA is unavailable.

### 3. Build and validate two compact versions

```bash
python compression/compress_dataset.py \
  --scenario LiveCodeBench \
  --train-data data/LiveCodeBench/LiveCodeBench_train.pkl \
  --model-path runs/LiveCodeBench/irt_model_run1/ \
  --test-data data/LiveCodeBench/LiveCodeBench_test.pkl \
  --ratios 0.5 0.75 \
  --output-dir runs/LiveCodeBench/compressed
```

A compression ratio is the fraction of samples removed. `0.5` keeps about half of the benchmark, while `0.75` keeps about one quarter.

### 4. Read the outputs

| Output | Meaning |
| --- | --- |
| `LiveCodeBench_anchor_ratio_0p5.pkl` | Selected sample indices and weights for the larger compact version |
| `LiveCodeBench_anchor_ratio_0p75.pkl` | Selected sample indices and weights for the smaller compact version |
| `LiveCodeBench_eval_ratio_*.json` | MAE, RMSE, Spearman correlation, and per-model errors on held-out models |

One seed is enough for a pipeline check. For a formal stability study, repeat a fixed seed list and aggregate the results.

<a id="included-examples"></a>

## 🧪 Included Examples

| Setting | Benchmark | Notebook | Prepared data |
| --- | --- | --- | --- |
| Text | LiveCodeBench | [`notebooks/text_livecodebench.ipynb`](notebooks/text_livecodebench.ipynb) | [`data/LiveCodeBench/`](data/LiveCodeBench/) |
| Multimodal | MathVista_MINI | [`notebooks/multimodal_mathvista_mini.ipynb`](notebooks/multimodal_mathvista_mini.ipynb) | [`data/MathVista_MINI/`](data/MathVista_MINI/) |
| Agent | SWE-bench Verified | [`notebooks/agentic_swe_bench_verified.ipynb`](notebooks/agentic_swe_bench_verified.ipynb) | [`data/swe_bench_verified/`](data/swe_bench_verified/) |

Use the notebooks to inspect the complete process. Use the prepared `.pkl` files when you only need to test the training and selection stages.

<a id="compress-a-new-benchmark"></a>

## 🧩 Compress a New Benchmark

You need item-level evaluation results from a small and capability-diverse set of models. Held-out model results are optional but strongly recommended for validating the final compact benchmark.

### Step 1: Prepare the response data

Train and test files use the same structure:

```python
{
    "models": ["model_a", "model_b", ...],
    "data": {
        "my_benchmark": {
            "correctness": np.ndarray  # [number of samples, number of models]
        }
    }
}
```

The `data` key must match the `--scenario` value. Train and test files must contain the same benchmark samples in the same order. Scores may be binary or normalized continuous values in `[0, 1]`. For continuous metrics, the standard pipeline binarizes the training records before synthesis, while the test file keeps the continuous scores; see [`docs/PIPELINE.md`](docs/PIPELINE.md) §1.4.

Six real models are enough for the train file, as long as their accuracy is roughly evenly spread rather than clustered at one level.

### Step 2: Add synthesized response patterns

The standard ZipBench setup starts from six real anchor models and synthesizes six additional records. These synthesized records broaden the observed capability range without running more real-model evaluations.

Raw text, multimodal, and agent logs have different formats. Follow [`docs/PIPELINE.md`](docs/PIPELINE.md) for the corresponding normalization and synthesis commands.

### Step 3: Train and select samples

Reuse the two commands from [Fastest Reproduction](#fastest-reproduction), replacing `LiveCodeBench` with your benchmark name and file paths.

The default representation settings are fixed across the release:

| Setting | Value |
| --- | ---: |
| Fingerprint dimension | 5 |
| Learning rate | 0.1 |
| Training epochs | 2,000 |
| Recommended removal ratios | 0.5 and 0.75 |

### Step 4: Validate and integrate

When `--test-data` is provided, `compress_dataset.py` compares compact and full scores on held-out models. Check:

- **MAE / RMSE:** how closely compact scores match full scores.
- **Spearman correlation:** how well model rankings are preserved.
- **Per-model errors:** whether a particular model is poorly estimated.

The output `.pkl` contains the selected indices and weights. A new model's estimated full score is the weighted average of its per-sample scores on those selected items.

## 🧠 Method Overview

1. **Initial evaluation:** collect item-level outcomes from about six anchor models with evenly spread accuracy.
2. **Response synthesis:** create additional model-response patterns across the observed accuracy range.
3. **Representation learning:** fit a multidimensional 2PL IRT model to learn model and sample fingerprints.
4. **Sample selection:** cluster sample fingerprints with K-means and retain one weighted representative per cluster.
5. **Validation:** compare compact and full benchmark scores on held-out models.

## 🗂️ Repository Structure

```text
zipbench/
├── data/             # prepared examples for text, multimodal, and agent tasks
├── data_prep/        # raw-record normalization, synthesis, and aggregation
├── compression/      # IRT training, sample selection, and validation
├── notebooks/        # end-to-end worked examples
├── py-irt/           # required patched dependency
├── docs/PIPELINE.md  # detailed commands for every record format
└── requirements.txt
```

## 🔧 Troubleshooting

| Problem | Likely cause and fix |
| --- | --- |
| `numpy._core` error while loading a `.pkl` | The file was saved with NumPy 2.x. Re-save it in a NumPy 1.x environment. |
| Spearman is undefined | Compression is too aggressive and all models receive the same score. Keep more samples. |
| Fewer synthesized records than requested | Some accuracy bins could not be filled from the real records. Keep the smaller synthetic set as-is; no extra synthesis is performed. |
| Training does not start on CPU | Add `--device cpu` and expect a slower run. |

## 📚 Citation

Please cite ZipBench using the BibTeX in the [main README](../README.md#citation).
