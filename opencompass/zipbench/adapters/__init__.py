"""Benchmark-specific ZipBench adapters.

The ZipBench core (``zipbench.dataset`` / ``zipbench.evaluator``) handles the
common case: one dataset row == one selection unit, scored by exact match. A
benchmark that breaks that shape (multi-step problems, code-execution scoring,
sub-row selection units, ...) lives here as a small adapter providing a
``ZipBench`` loader + a weighted evaluator, wired in purely through its manifest
(``--units-from <adapter>``) -- no core code becomes benchmark-aware.

Modules are imported lazily (by their fully-qualified ``type`` string inside the
run.py worker subprocesses), so importing this package stays cheap.
"""
