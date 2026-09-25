<div align="center">
  <img src="assets/logo.png" width="360" alt="ZipBench">
  <h3>A Compact Benchmark Zoo for Fast and Reliable Model Evaluation</h3>
  <p>100+ ready-to-use compact versions of text, multimodal, and agent benchmarks.</p>
  <p>
    <a href="https://arxiv.org/abs/2609.12475"><img src="https://img.shields.io/badge/Oral-EMNLP%202026-6f42c1" alt="Oral: EMNLP 2026"></a>
    <img src="https://img.shields.io/badge/ZipBench%20Zoo-100%2B%20Compact%20Benchmarks-b8860b" alt="100+ compact benchmarks">
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue" alt="Apache 2.0 License"></a>
  </p>
  <p>
    <a href="#quick-start">Quick Start</a> ·
    <a href="opencompass/README.md">Text Evaluation</a> ·
    <a href="VLMEvalKit/README.md">Multimodal Evaluation</a> ·
    <a href="agent/README.md">Agent Evaluation</a> ·
    <a href="zipbench/README.md">Build Your Own</a> ·
    <a href="#citation">Citation</a>
  </p>
</div>

This repository is the official codebase for our EMNLP paper "ZipBench: Low-Cost Framework for Compressing Comprehensive Benchmarks of Large Language Models".

**TL;DR.** ZipBench provides 100+ ready-to-use compact versions of text, multimodal, and agent benchmarks, cutting evaluation cost by 60–80% while closely preserving full-benchmark scores and model rankings.


## 🎉 News

2026-08 - Our "ZipBench" has been accepted for an **Oral presentation** at EMNLP'26! [[Paper]](https://arxiv.org/abs/2609.12475) 👈🎉Please read it！


## ✨ What You Can Do

| Your goal | Start here | What this repository provides |
| --- | --- | --- |
| Evaluate an LLM on text benchmarks | [`opencompass/`](opencompass/README.md) | Ready-to-run `full`, `small`, and `tiny` configs for 25 benchmark families |
| Evaluate a VLM on multimodal benchmarks | [`VLMEvalKit/`](VLMEvalKit/README.md) | A simple `--subset full\|small\|tiny` switch for 13 benchmark families |
| Evaluate an agent on interactive benchmarks | [`agent/`](agent/README.md) | Compact task lists and weighted scoring for 9 benchmarks, used with their official evaluation tools |
| Compress a new benchmark | [`zipbench/`](zipbench/README.md) | The complete construction pipeline from item-level results to a weighted compact benchmark |
| Reproduce representative examples | [`zipbench/notebooks/`](zipbench/notebooks/) | End-to-end examples for text, multimodal, and agent benchmarks |

The paper builds **ZipBench Zoo**, containing more than 100 compact benchmark versions across text, multimodal, and agent tasks. This repository provides ready-to-use evaluation for all three settings, together with the full benchmark-construction pipeline.

<a id="quick-start"></a>

## 🚀 Quick Start

> [!IMPORTANT]
> `opencompass/`, `VLMEvalKit/`, `agent/`, and `zipbench/` use different dependency stacks. Create a separate environment for each folder.

### 📝 Evaluate a text LLM

```bash
cd opencompass
python run.py zipbench/configs/eval_arc_challenge_small.py
```

The command evaluates the model configured inside the file on the **small** ARC-Challenge benchmark. See the [text evaluation guide](opencompass/README.md) for installation and model configuration.

### 🖼️ Evaluate a vision-language model

```bash
cd VLMEvalKit
python run.py \
  --data MMStar \
  --model Qwen2.5-VL-7B-Instruct \
  --subset small
```

Change `small` to `tiny` for a lower-cost run or to `full` for the complete benchmark. See the [multimodal evaluation guide](VLMEvalKit/README.md) for setup and supported models.

### 🤖 Evaluate an agent

Agent benchmarks use different official evaluation tools. ZipBench supplies the selected task IDs and weights, while each official tool still runs and scores the agent normally.

```bash
cd agent
pip install -e .
zipbench-agent list
```

Choose a benchmark, export its `small` or `tiny` task list, run the official evaluation tool, and pass its results back to `zipbench-agent score`. See the [agent evaluation guide](agent/README.md) for complete commands.

### 🧩 Compress your own benchmark

```bash
cd zipbench
python compression/train_irt.py \
  --scenario LiveCodeBench \
  --train-pkl data/LiveCodeBench/LiveCodeBench_train.pkl \
  --output-dir runs/LiveCodeBench \
  --device cuda --seed 7 --run-id 1
```

Follow the [ZipBench construction guide](zipbench/README.md) for a two-command reproduction and the full workflow for a new benchmark.

## 🎯 Which Version Should I Use?

| Version | Samples used on average | Fidelity to the full benchmark | Recommended use |
| --- | ---: | --- | --- |
| `full` | 100% | Exact official evaluation | Final reporting and leaderboards |
| `small` | 42.6% | MAE 0.008, Spearman 0.99 | Routine development and model comparison |
| `tiny` | 22.5% | MAE 0.015, Spearman 0.98 | Fast screening, ablations, and repeated runs |

These values are averages across ZipBench Zoo. The actual size of each compact benchmark is selected separately according to its redundancy and evaluation fidelity.

> [!NOTE]
> Scores from `small` and `tiny` are **weighted estimates** of the corresponding full-benchmark scores. No manual rescaling is needed in the provided runners.

## 📦 Benchmark Coverage

| Track | Examples | Ready-to-run evaluation |
| --- | --- | --- |
| **Text** | MMLU, MMLU-Pro, GSM8K, BBH, LongBench, HumanEval+, LiveCodeBench, and more | [`opencompass/`](opencompass/README.md) |
| **Multimodal** | MMMU, MMMU-Pro, MMStar, MathVista, OCRBench v2, OmniDocBench v1.5, and more | [`VLMEvalKit/`](VLMEvalKit/README.md) |
| **Agent** | SWE-bench, Terminal-Bench, τ³-bench, OSWorld-Verified, Toolathlon, and more | [`agent/`](agent/README.md) |

Complete benchmark names and sample counts are listed in the three evaluation guides.

## 🧠 How ZipBench Works

You do not need to run the construction pipeline to use the released `small` and `tiny` benchmarks. For readers who want to reproduce the method, ZipBench uses four steps:

1. **Evaluate a few diverse models.** The standard setup uses six anchor LLMs with different capability levels.
2. **Expand the observed behavior.** Additional response patterns are synthesized without evaluating more real models.
3. **Learn which samples are informative.** ZipBench learns a compact representation of how models behave on each sample.
4. **Select and weight representative samples.** Similar samples are grouped, and one weighted representative is retained from each group.

The implementation uses multidimensional 2PL item response theory and K-means clustering. The paper provides error and rank-consistency analyses for the resulting compact benchmarks.

## 🗂️ Repository Structure

```text
ZipBench/
├── assets/          # Logo and repository assets
├── opencompass/     # Text evaluation on full, small, or tiny benchmarks
├── VLMEvalKit/      # Multimodal evaluation on full, small, or tiny benchmarks
├── agent/           # Agent evaluation through official benchmark tools
└── zipbench/        # Benchmark compression and reproducibility pipeline
```

The original upstream documentation is retained as [`opencompass/README_OpenCompass.md`](opencompass/README_OpenCompass.md) and [`VLMEvalKit/README_VLMEvalKit.md`](VLMEvalKit/README_VLMEvalKit.md).

<a id="citation"></a>

## 📚 Citation

```bibtex
@misc{huang2026zipbenchlowcostframeworkcompressing,
      title={Zipbench: Low-Cost Framework for Compressing Comprehensive Benchmarks of Large Language Models}, 
      author={Zhongzhan Huang and Junxin Li and Guoming Ling and Yupei Lin and Shanshan Zhong and Hefeng Wu},
      year={2026},
      eprint={2609.12475},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2609.12475}, 
}
```

## 🙏 Acknowledgements

ZipBench builds on [OpenCompass](https://github.com/open-compass/opencompass), [VLMEvalKit](https://github.com/open-compass/VLMEvalKit), and [py-irt](https://github.com/nd-ball/py-irt). We thank their maintainers and the authors of all supported benchmarks.

This repository is released under the [Apache 2.0 License](LICENSE). The bundled `py-irt` code retains its original MIT license.
