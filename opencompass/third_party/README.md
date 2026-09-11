# Third-party vendored code

## livecodebench/

Vendored copy of the official [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench)
repository, pinned at commit `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24` (MIT
license, see `livecodebench/LICENSE`). The code is unmodified.

It is vendored (rather than fetched at install time) so that the exact
dataset-loading, prompt, and grading logic used by
`opencompass/datasets/livecodebench/official.py` stays fixed: LiveCodeBench
subset scores are only comparable under this pinned harness. To upgrade,
replace the directory with a checkout of the new upstream commit and update
the pin recorded here.
