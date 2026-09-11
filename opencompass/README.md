<div align="center">
  <a href="../README.md"><img src="../assets/logo.png" width="360" alt="ZipBench"></a>
  <h2>📝 Text Benchmark Evaluation</h2>
  <p>Evaluate LLMs on MMLU, GSM8K, BBH, LongBench, and more with fewer samples.</p>
  <p>
    <img src="https://img.shields.io/badge/Based%20on-OpenCompass-2f6f9f" alt="Based on OpenCompass">
    <img src="https://img.shields.io/badge/Text-25%20Benchmark%20Families-b8860b" alt="25 text benchmark families">
    <img src="https://img.shields.io/badge/Evaluation-full%20%7C%20small%20%7C%20tiny-6f42c1" alt="full small tiny evaluation">
  </p>
  <p>
    <a href="#quick-start">Quick Start</a> ·
    <a href="#supported-benchmarks">Benchmarks</a> ·
    <a href="#use-your-own-model">Use Your Model</a> ·
    <a href="../agent/README.md">Agent Evaluation</a> ·
    <a href="../zipbench/README.md">Compress a Benchmark</a> ·
    <a href="../README.md">Back to ZipBench</a>
  </p>
</div>

This folder is for **text-only LLM evaluation**. It extends OpenCompass with ready-to-use **`small`** and **`tiny`** versions of common benchmarks.

Instead of evaluating every question in a benchmark, ZipBench evaluates only selected representative questions and automatically combines their scores using learned weights. The final result is an estimate of the score on the complete benchmark.

| Mode | What is evaluated | Best for |
| --- | --- | --- |
| `full` | Every original benchmark sample | Exact final evaluation |
| `small` | A fidelity-first weighted subset | Routine development and comparison |
| `tiny` | A more aggressively compressed weighted subset | Fast screening and repeated experiments |

> [!TIP]
> For normal use, start with `small`. You do not need to run the benchmark-compression pipeline yourself.

<a id="quick-start"></a>

## 🚀 Quick Start

### 1. Install OpenCompass for ZipBench

```bash
conda create -n zipbench-text python=3.10 -y
conda activate zipbench-text
cd opencompass
pip install -e ".[vllm]"
```

For API-based models, use `pip install -e ".[api]"` instead.

### 2. Select a model

Each benchmark file imports an OpenCompass model config. **Check this import before running**, because several research configs point to large models.

For example, a lightweight bundled config is:

```python
with read_base():
    from opencompass.configs.models.qwen3.vllm_qwen3_4b_instruct import models
```

To use local model weights:

```bash
export OC_MODEL_ROOT=/path/to/model/weights
```

Without `OC_MODEL_ROOT`, the model config may download weights from Hugging Face.

### 3. Run `small`, `tiny`, or `full`

```bash
# Recommended: 321 selected ARC-Challenge questions
python run.py zipbench/configs/eval_arc_challenge_small.py

# Faster: 148 selected questions
python run.py zipbench/configs/eval_arc_challenge_tiny.py

# Exact: all 1,172 questions
python run.py zipbench/configs/eval_arc_challenge.py
```

Results are written under `outputs/`. For `small` and `tiny`, the reported score is already the weighted estimate of the full-benchmark score.

A configuration can be checked without launching inference:

```bash
python run.py zipbench/configs/eval_arc_challenge_small.py --dry-run
```

## 🧭 What ZipBench Changes

| Kept from OpenCompass | Added by ZipBench |
| --- | --- |
| Model backends and model configs | `small` and `tiny` sample selections |
| Benchmark prompts and official-style scoring | Automatic weighted score aggregation |
| Inference, logging, and output structure | Validation that local data match the released subset |
| Full-benchmark evaluation | Directly comparable `full`, `small`, and `tiny` runs |

The original OpenCompass documentation is preserved in [`README_OpenCompass.md`](README_OpenCompass.md).

<a id="supported-benchmarks"></a>

## 📦 Supported Benchmarks

This release contains **27 compact benchmark specifications across 25 text benchmark families**. OpenBookQA and SciCode each contain two separately compressed variants.

<details>
<summary><strong>View all benchmark sizes and configuration files</strong></summary>

| Benchmark | Full | Small | Tiny | Config |
| --- | ---: | ---: | ---: | --- |
| ARC-Challenge | 1,172 | 321 | 148 | [`eval_arc_challenge.py`](zipbench/configs/eval_arc_challenge.py) |
| Arena-Hard | 500 | 250 | 123 | [`eval_arenahard.py`](zipbench/configs/eval_arenahard.py) |
| BBH | 5,761 | 725 | 365 | [`eval_bbh.py`](zipbench/configs/eval_bbh.py) |
| C3 | 1,825 | 913 | 59 | [`eval_c3.py`](zipbench/configs/eval_c3.py) |
| CommonsenseQA | 1,221 | 611 | 258 | [`eval_commonsenseqa.py`](zipbench/configs/eval_commonsenseqa.py) |
| GPQA | 1,192 | 503 | 294 | [`eval_gpqa.py`](zipbench/configs/eval_gpqa.py) |
| GSM8K | 1,319 | 484 | 320 | [`eval_gsm8k.py`](zipbench/configs/eval_gsm8k.py) |
| HellaSwag | 10,042 | 951 | 147 | [`eval_hellaswag.py`](zipbench/configs/eval_hellaswag.py) |
| HumanEval+ | 164 | 82 | 31 | [`eval_humaneval_plus.py`](zipbench/configs/eval_humaneval_plus.py) |
| IFEval | 541 | 260 | 136 | [`eval_ifeval.py`](zipbench/configs/eval_ifeval.py) |
| LiveCodeBench | 1,055 | 528 | 265 | [`eval_livecodebench.py`](zipbench/configs/eval_livecodebench.py) |
| LongBench | 4,750 | 885 | 151 | [`eval_longbench.py`](zipbench/configs/eval_longbench.py) |
| LongBench v2, 0-shot | 503 | 220 | 126 | [`eval_longbenchv2.py`](zipbench/configs/eval_longbenchv2.py) |
| MATH, hard split | 1,324 | 301 | 237 | [`eval_math.py`](zipbench/configs/eval_math.py) |
| MBPP+ | 399 | 200 | 88 | [`eval_mbpp_plus.py`](zipbench/configs/eval_mbpp_plus.py) |
| MMLU | 14,042 | 767 | 446 | [`eval_mmlu.py`](zipbench/configs/eval_mmlu.py) |
| MMLU-Pro | 12,032 | 388 | 259 | [`eval_mmlu_pro.py`](zipbench/configs/eval_mmlu_pro.py) |
| MuSR | 756 | 331 | 190 | [`eval_musr.py`](zipbench/configs/eval_musr.py) |
| OpenBookQA | 500 | 244 | 170 | [`eval_obqa.py`](zipbench/configs/eval_obqa.py) |
| OpenBookQA with fact | 500 | 219 | 125 | [`eval_obqa.py`](zipbench/configs/eval_obqa.py) |
| SciBench | 583 | 272 | 146 | [`eval_scibench.py`](zipbench/configs/eval_scibench.py) |
| SciCode | 288 | 167 | 70 | [`eval_scicode.py`](zipbench/configs/eval_scicode.py) |
| SciCode with background | 288 | 144 | 72 | [`eval_scicode.py`](zipbench/configs/eval_scicode.py) |
| SimpleQA | 4,326 | 1,085 | 274 | [`eval_simpleqa.py`](zipbench/configs/eval_simpleqa.py) |
| TheoremQA | 800 | 400 | 201 | [`eval_theoremqa.py`](zipbench/configs/eval_theoremqa.py) |
| TruthfulQA | 817 | 203 | 144 | [`eval_truthfulqa.py`](zipbench/configs/eval_truthfulqa.py) |
| WinoGrande | 1,267 | 318 | 159 | [`eval_winogrande.py`](zipbench/configs/eval_winogrande.py) |

</details>

The released sample identifiers, weights, and data checks are stored under [`zipbench/subsets/`](zipbench/subsets/).

<a id="use-your-own-model"></a>

## 🧩 Use Your Own Model

Use an existing model config under [`opencompass/configs/models/`](opencompass/configs/models/) or add a new one, then update the `models` import in the selected benchmark file.

Two practical rules:

1. Keep generation settings such as `temperature`, `max_out_len`, and tensor parallelism in the model config.
2. Give reasoning models enough output length. Truncated reasoning can silently reduce the final score.

For complete model-backend instructions, see the [OpenCompass model guide](https://opencompass.readthedocs.io/en/latest/user_guides/models.html).

## ⚙️ Data and Optional Dependencies

Dataset loading follows OpenCompass. Some datasets use the OpenCompass data pack, while others are downloaded from their official or Hugging Face sources on first use.

<details>
<summary><strong>Optional scoring dependencies</strong></summary>

| Benchmark | Additional requirement |
| --- | --- |
| IFEval | `pip install langdetect` and `python -c "import nltk; nltk.download('punkt')"` |
| HumanEval+ and MBPP+ | `pip install evalplus` |
| TheoremQA | `pip install latex2sympy2_extended` |
| SciCode | Unzip [scicode.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/scicode.zip) into `data/scicode/` and [scicode_test_data.zip](https://github.com/open-compass/storage/releases/download/v0.1.0/scicode_test_data.zip) into `data/scicode/test_data/`, then run `python zipbench/make_scicode_with_background.py` to build the with-background file. |
| LiveCodeBench | The dataset is downloaded from [livecodebench/code_generation_lite](https://huggingface.co/datasets/livecodebench/code_generation_lite) on first run. |
| Arena-Hard and SimpleQA | Scored by an LLM judge. The judge model is configurable in each config file. |

</details>

If a validation error reports missing or changed samples, the local benchmark version does not match the version used to build the released compact benchmark. Use the expected dataset version instead of evaluating an incomplete set.

## 🗂️ Key Files

```text
opencompass/
├── zipbench/configs/         # full, small, and tiny evaluation configs
├── zipbench/subsets/         # selected sample IDs, weights, and manifests
├── opencompass/configs/models/ # model configurations
├── outputs/                  # evaluation outputs
└── README_OpenCompass.md     # original upstream documentation
```

## 📚 Citation and Upstream Project

Please cite ZipBench using the BibTeX in the [main README](../README.md#citation).

This fork builds on [OpenCompass](https://github.com/open-compass/opencompass) and retains its Apache 2.0 license.
