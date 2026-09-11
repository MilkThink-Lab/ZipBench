<div align="center">
  <a href="../README.md"><img src="../assets/logo.png" width="360" alt="ZipBench"></a>
  <h2>🤖 Agent Benchmark Evaluation</h2>
  <p>Evaluate agents on SWE-bench, Terminal-Bench, τ³-bench, OSWorld, and more with fewer tasks.</p>
  <p>
    <img src="https://img.shields.io/badge/Uses-Official%20Benchmark%20Tools-2f6f9f" alt="Uses official benchmark tools">
    <img src="https://img.shields.io/badge/Agent-9%20Benchmarks-b8860b" alt="9 agent benchmarks">
    <img src="https://img.shields.io/badge/Evaluation-full%20%7C%20small%20%7C%20tiny-6f42c1" alt="full small tiny evaluation">
  </p>
  <p>
    <a href="#quick-start">Quick Start</a> ·
    <a href="#supported-benchmarks">Benchmarks</a> ·
    <a href="#how-agent-evaluation-works">How It Works</a> ·
    <a href="#scoring-and-missing-tasks">Scoring</a> ·
    <a href="../zipbench/README.md">Compress a Benchmark</a> ·
    <a href="../README.md">Back to ZipBench</a>
  </p>
</div>

An **agent benchmark** tests whether a model can complete multi-step tasks such as fixing a repository, operating a terminal, using tools, or interacting with a computer interface. Each benchmark has its own official evaluation tool, often called a **harness**, which runs the tasks and records whether the agent succeeded.

Running every task can be slow and expensive. ZipBench provides **`small`** and **`tiny`** task lists that retain representative tasks from the full benchmark. After the official harness finishes, ZipBench combines the task results with learned weights to estimate the full-benchmark score.

| Mode | What is evaluated | Best for |
| --- | --- | --- |
| `full` | Every task, using the official harness normally | Exact final evaluation |
| `small` | A fidelity-first weighted task subset | Routine development and comparison |
| `tiny` | A more aggressively compressed weighted subset | Fast screening and repeated experiments |

> [!IMPORTANT]
> ZipBench does not replace or modify the official harness. Use the same agent, model, environment, and evaluation settings that you would use for a full run. ZipBench only supplies the selected task IDs and reconstructs the weighted score.

<a id="quick-start"></a>

## 🚀 Quick Start

### 1. Install the lightweight ZipBench helper

```bash
conda create -n zipbench-agent python=3.10 -y
conda activate zipbench-agent
cd agent
pip install -e .
zipbench-agent list
```

The helper has no third-party Python dependencies. Install the official benchmark harness separately by following the guide for the benchmark you want to run.

### 2. Export a compact task list

```bash
# Print the task-filter arguments for Terminal-Bench small
zipbench-agent tasks terminal_bench --subset small --format args
```

The output contains only task IDs, not benchmark data or model responses. Different harnesses accept IDs in different forms, so ZipBench can export them as plain lines, command-line arguments, JSON, or benchmark-specific metadata.

### 3. Run the official harness

For example, a Terminal-Bench run in a Bash-compatible shell is:

```bash
harbor run --dataset terminal-bench@2.0 --agent <agent> --model <model> \
  --job-name my-small-run --jobs-dir jobs \
  $(zipbench-agent tasks terminal_bench --subset small --format args)
```

Replace `<agent>` and `<model>` with the names expected by Terminal-Bench. Other benchmarks use different commands; copy the exact command from the corresponding [benchmark guide](#supported-benchmarks).

### 4. Estimate the full-benchmark score

```bash
zipbench-agent score terminal_bench --subset small \
  --results jobs/my-small-run
```

The command checks that every selected task has a result, reads the harness's native output, and reports the weighted estimate. No manual rescaling is needed.

<a id="how-agent-evaluation-works"></a>

## 🧭 How Agent Evaluation Works

<div align="center">
  <b>ZipBench task IDs</b> &nbsp;→&nbsp; <b>official harness</b> &nbsp;→&nbsp; <b>native task results</b> &nbsp;→&nbsp; <b>weighted score</b>
</div>

| Kept from the official benchmark | Added by ZipBench |
| --- | --- |
| Agent and model setup | Released `small` and `tiny` task IDs |
| Task environments and execution | A task-filter format accepted by the harness |
| Prompts, tools, and official per-task scoring | Learned weights for selected tasks |
| Full-benchmark evaluation | Result parsing and full-score estimation |

This design keeps compact and full runs directly comparable: both use the same official evaluator, while only the number of tasks changes.

<a id="supported-benchmarks"></a>

## 📦 Supported Benchmarks

This release contains **9 compact agent benchmark configurations**.

| Benchmark | Full | Small | Tiny | Guide |
| --- | ---: | ---: | ---: | --- |
| SWE-bench Verified | 500 | 188 | 125 | [SWE-bench](docs/benchmarks/swe_bench.md) |
| SWE-bench Multilingual | 300 | 110 | 78 | [SWE-bench](docs/benchmarks/swe_bench.md) |
| Terminal-Bench 2.0 | 89 | 72 | 50 | [Terminal-Bench](docs/benchmarks/terminal_bench.md) |
| τ³-bench telecom | 114 | 52 | 22 | [τ³-bench](docs/benchmarks/tau2.md) |
| τ³-bench retail | 114 | 79 | 72 | [τ³-bench](docs/benchmarks/tau2.md) |
| τ³-bench airline | 50 | 23 | 14 | [τ³-bench](docs/benchmarks/tau2.md) |
| SWE-Bench Pro, public set | 731 | 366 | 160 | [SWE-Bench Pro](docs/benchmarks/swe_bench_pro.md) |
| OSWorld-Verified | 361 | 205 | 98 | [OSWorld-Verified](docs/benchmarks/osworld_verified.md) |
| Toolathlon | 108 | 76 | 58 | [Toolathlon](docs/benchmarks/toolathlon.md) |

Each guide specifies the supported upstream version, task-filter command, accepted result files, and any benchmark-specific scoring details.

The released task IDs, weights, and manifests are stored under [`zipbench_agent/subsets_data/`](zipbench_agent/subsets_data/).

## 🛠️ Useful Commands

```bash
# List benchmark keys, sizes, and metrics
zipbench-agent list

# Save one task ID per line
zipbench-agent tasks swe_bench_verified --subset tiny > tiny_ids.txt

# Export a JSON task list
zipbench-agent tasks toolathlon --subset tiny --format json

# Produce a machine-readable score report
zipbench-agent score tau2_airline --subset tiny \
  --results results.json --json
```

For harnesses that accept an inline task filter, `zipbench-agent run` can append the filter and launch the command:

```bash
zipbench-agent run tau2_airline --subset tiny -- \
  tau2 run --domain airline --agent-llm <model> --user-llm <model>
```

Some harnesses require a task-list or metadata file instead. Use the benchmark guide rather than `run` in those cases.

<a id="scoring-and-missing-tasks"></a>

## 🎯 Scoring and Missing Tasks

For each selected task, the official harness produces a score between 0 and 1. ZipBench multiplies each task score by its released weight and sums the results. The weights add up to 1, so the output is already on the same scale as the full benchmark's average score.

By default, `zipbench-agent score` stops if a selected task is missing. It distinguishes between:

- **Absent tasks:** no result was found, usually because the task never ran.
- **Errored tasks:** the harness recorded the task but did not produce a valid score.

Run or repair missing tasks before reporting a score. If a task genuinely failed and should count as zero, pass `--missing zero`; ZipBench keeps its original weight and does not renormalize the remaining tasks.

> [!TIP]
> Start with `small` for normal comparisons. Use `tiny` for quick iteration, and run the official full benchmark when an exact leaderboard score is required.

## 🗂️ Key Files

```text
agent/
├── zipbench_agent/                 # task export, result parsing, and scoring
│   └── subsets_data/               # selected task IDs, weights, and manifests
├── docs/benchmarks/                # commands for each official harness
├── tests/                          # CLI, adapter, and scoring tests
├── tools/                          # subset conversion utilities
└── pyproject.toml                  # lightweight CLI package
```

## 📚 Citation and Upstream Projects

Please cite ZipBench using the BibTeX in the [main README](../README.md#citation).

Agent evaluation uses the official tools and scoring conventions of each supported benchmark. Please also cite the benchmark and harness used in your experiment.
