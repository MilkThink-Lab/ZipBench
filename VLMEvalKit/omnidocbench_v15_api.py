#!/usr/bin/env python
"""Deprecated compatibility shim for the old API-only OmniDocBench evaluator.

The evaluator now lives in omnidocbench_v15.py (package omnidocbench_v15/)
with both API and vLLM inference backends. This shim keeps existing
invocations working by injecting `--backend api` into run subcommands.
"""

from __future__ import annotations

import os
import runpy
import sys

# "import omnidocbench_v15" would resolve to the package directory, so the
# CLI script of the same name must be executed by path instead.
_CLI_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "omnidocbench_v15.py")


def main() -> None:
    print(
        "[omnidocbench_v15_api.py] Deprecated: use omnidocbench_v15.py instead "
        "(this shim forwards with --backend api).",
        file=sys.stderr,
    )
    argv = sys.argv[1:]
    if argv and argv[0] in {"run", "run_and_eval"} and "--backend" not in argv:
        argv = [argv[0], "--backend", "api"] + argv[1:]
    sys.argv = [_CLI_PATH] + argv
    runpy.run_path(_CLI_PATH, run_name="__main__")


if __name__ == "__main__":
    main()
