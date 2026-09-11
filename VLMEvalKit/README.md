<div align="center">
  <a href="../README.md"><img src="../assets/logo.png" width="360" alt="ZipBench"></a>
  <h2>🖼️ Multimodal Benchmark Evaluation</h2>
  <p>Evaluate VLMs on MMStar, MMMU, MathVista, OCRBench, and more with fewer samples.</p>
  <p>
    <img src="https://img.shields.io/badge/Based%20on-VLMEvalKit-2f6f9f" alt="Based on VLMEvalKit">
    <img src="https://img.shields.io/badge/Multimodal-13%20Benchmark%20Families-b8860b" alt="13 multimodal benchmark families">
    <img src="https://img.shields.io/badge/Evaluation-full%20%7C%20small%20%7C%20tiny-6f42c1" alt="full small tiny evaluation">
  </p>
  <p>
    <a href="#quick-start">Quick Start</a> ·
    <a href="#supported-benchmarks">Benchmarks</a> ·
    <a href="#special-runners">Special Runners</a> ·
    <a href="#use-your-own-model">Use Your Model</a> ·
    <a href="../agent/README.md">Agent Evaluation</a> ·
    <a href="../README.md">Back to ZipBench</a>
  </p>
</div>

This folder is for **vision-language model evaluation**. It extends VLMEvalKit with ready-to-use **`small`** and **`tiny`** versions of common multimodal benchmarks.

The model, prompt, inference, and benchmark-specific scoring remain unchanged. ZipBench only selects fewer representative samples and automatically applies their weights to estimate the complete benchmark score.

| Mode | What is evaluated | Best for |
| --- | --- | --- |
| `full` | Every original benchmark sample | Exact final evaluation |
| `small` | A fidelity-first weighted subset | Routine development and comparison |
| `tiny` | A more aggressively compressed weighted subset | Fast screening and repeated experiments |

> [!TIP]
> In most cases, using ZipBench only requires adding `--subset small` to a normal VLMEvalKit command.

<a id="quick-start"></a>

## 🚀 Quick Start

### 1. Install VLMEvalKit for ZipBench

```bash
conda create -n zipbench-vlm python=3.10 -y
conda activate zipbench-vlm
cd VLMEvalKit
pip install -e .
```

Install vLLM separately when using a local vLLM backend. Keep this environment separate from `opencompass/` and `zipbench/`.

VLMEvalKit stores benchmark data in `~/LMUData` by default. To use another location:

```bash
export LMUData=/path/to/LMUData
```

### 2. Check the model name

```bash
vlmutil mlist all
```

Models are registered in [`vlmeval/config.py`](vlmeval/config.py). For a local checkpoint, update the selected entry's `model_path`.

### 3. Run `small`, `tiny`, or `full`

```bash
# Recommended compact evaluation
python run.py \
  --data MMStar \
  --model Qwen2.5-VL-7B-Instruct \
  --subset small

# Faster evaluation
python run.py \
  --data MMStar \
  --model Qwen2.5-VL-7B-Instruct \
  --subset tiny

# Complete benchmark
python run.py \
  --data MMStar \
  --model Qwen2.5-VL-7B-Instruct \
  --subset full
```

Outputs are written under `outputs/<model>/`. Compact runs include `_ZIP_small` or `_ZIP_tiny` in their file names, so they do not overwrite full-benchmark results. The reported score is already weighted.

## 🧭 What ZipBench Changes

| Kept from VLMEvalKit | Added by ZipBench |
| --- | --- |
| Model wrappers and API backends | `--subset full\|small\|tiny` |
| Prompts, generation, and benchmark scoring | Selected sample IDs and learned weights |
| Output files and reuse workflow | Separate names for compact-run outputs |
| Full-benchmark evaluation | Validation that local data match the released subset |

The original VLMEvalKit documentation is preserved in [`README_VLMEvalKit.md`](README_VLMEvalKit.md).

## ⚙️ Data and Optional Dependencies

Dataset loading follows VLMEvalKit. Most benchmarks are downloaded into `LMUData` on first use. MMMU-Pro and OmniDocBench v1.5 are scored by their official evaluation code, which the dedicated runners fetch automatically.

<details>
<summary><strong>Optional scoring dependencies</strong></summary>

| Benchmark | Additional requirement |
| --- | --- |
| MathVista_MINI, LogicVista, and SimpleVQA | Scored by an LLM judge through an OpenAI-compatible API. Set `OPENAI_API_KEY` and `OPENAI_API_BASE`. |
| MMMU_Pro_10c and MMMU_Pro_V | The official [MMMU](https://github.com/MMMU-Benchmark/MMMU) repository is cloned into `third_party/` on first run. |
| OmniDocBench v1.5 | Download the GT JSON and page images from [opendatalab/OmniDocBench](https://huggingface.co/datasets/opendatalab/OmniDocBench). The official repository is cloned into `third_party/` on first run and evaluated in Docker: `docker pull sunyuefeng/omnidocbench-env:v1.5`. |

</details>

<a id="supported-benchmarks"></a>

## 📦 Supported Benchmarks

This release contains **18 compact benchmark specifications across 13 multimodal benchmark families**. OmniDocBench v1.5 is compressed separately for six official metrics.

<details>
<summary><strong>View all benchmark sizes and runners</strong></summary>

| Benchmark | Full | Small | Tiny | Runner |
| --- | ---: | ---: | ---: | --- |
| CountBenchQA | 487 | 244 | 92 | `run.py` |
| LogicVista | 447 | 224 | 112 | `run.py` |
| MathVista_MINI | 1,000 | 376 | 251 | `run.py` |
| MMMU_DEV_VAL | 1,050 | 460 | 255 | `run.py` |
| MMMU_Pro_10c | 1,730 | 650 | 429 | [`mmmu_pro.py`](mmmu_pro.py) |
| MMMU_Pro_V | 1,730 | 866 | 434 | [`mmmu_pro.py`](mmmu_pro.py) |
| MMStar | 1,500 | 540 | 376 | `run.py` |
| MMVP | 300 | 226 | 150 | `run.py` |
| OCRBench_v2 | 10,000 | 5,005 | 2,508 | `run.py` |
| RealWorldQA | 765 | 431 | 96 | `run.py` |
| SimpleVQA | 2,025 | 1,014 | 445 | `run.py` |
| SpatialEval | 4,635 | 1,161 | 293 | `run.py` |
| OmniDocBench v1.5 | Six metric-specific versions | See below | See below | [`omnidocbench_v15.py`](omnidocbench_v15.py) |

**OmniDocBench v1.5**

| Official metric | Full | Small | Tiny |
| --- | ---: | ---: | ---: |
| Text edit distance | 1,290 | 484 | 162 |
| Table edit distance | 351 | 176 | 88 |
| Formula edit distance | 200 | 75 | 50 |
| Reading-order edit distance | 1,290 | 485 | 323 |
| Formula CDM | 200 | 100 | 50 |
| Table TEDS | 512 | 256 | 128 |

</details>

The released sample identifiers, weights, and data checks are stored under [`vlmeval/zipbench/subsets/`](vlmeval/zipbench/subsets/).

<a id="special-runners"></a>

## 🛠️ Special Runners

Most benchmarks use `run.py`. MMMU-Pro and OmniDocBench v1.5 use dedicated scripts to stay aligned with their official evaluation code.

### MMMU-Pro

```bash
python mmmu_pro.py run_and_eval \
  --backend vllm \
  --model /path/to/model \
  --dataset MMMU_Pro_10c \
  --subset tiny
```

Use `--dataset MMMU_Pro_V` for the vision-only variant. API models can be selected with `--backend api --model-name <name>`.

### OmniDocBench v1.5

```bash
python omnidocbench_v15.py run_and_eval \
  --backend vllm \
  --model /path/to/model \
  --gt-json /path/to/OmniDocBench.json \
  --image-root /path/to/page_images \
  --subset tiny
```

The script evaluates the required page union once and then reports each of the six weighted official metrics separately. Run `python omnidocbench_v15.py run_and_eval --help` for backend and evaluator options.

<a id="use-your-own-model"></a>

## 🧩 Use Your Own Model

Add or update an entry in [`vlmeval/config.py`](vlmeval/config.py), then pass its registered name through `--model`.

Useful options:

- `--reuse` reuses compatible prediction files from an earlier run.
- `--judge <name>` selects an LLM judge when required by a benchmark.
- `--use-vllm --batch-size N` enables batched generation for supported local model wrappers.
- `SPLIT_THINK=1` removes `<think>...</think>` from the scored answer while keeping the reasoning in a separate output column.

API-based models and judges read credentials from `OPENAI_API_KEY` and `OPENAI_API_BASE` when applicable.

## 🗂️ Key Files

```text
VLMEvalKit/
├── run.py                         # standard full, small, and tiny evaluation
├── mmmu_pro.py                    # dedicated MMMU-Pro runner
├── omnidocbench_v15.py            # dedicated OmniDocBench v1.5 runner
├── vlmeval/config.py              # model registry
├── vlmeval/zipbench/subsets/      # selected sample IDs, weights, and manifests
├── outputs/                       # evaluation outputs
└── README_VLMEvalKit.md           # original upstream documentation
```

## 📚 Citation and Upstream Project

Please cite ZipBench using the BibTeX in the [main README](../README.md#citation).

This fork builds on [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) and retains its Apache 2.0 license.
