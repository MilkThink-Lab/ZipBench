"""Registry mapping dataset keys to harness adapters."""

from .base import (  # noqa: F401
    Adapter, ParseResult, ResultParseError, TaskArgsUnsupportedError)
from .osworld import OSWorldAdapter
from .swe_bench import SweBenchAdapter
from .swe_bench_pro import SweBenchProAdapter
from .tau2 import Tau2Adapter
from .terminal_bench import TerminalBenchAdapter
from .toolathlon import ToolathlonAdapter

ADAPTERS = {
    "swe_bench_verified": SweBenchAdapter("swe_bench_verified"),
    "swe_bench_multilingual": SweBenchAdapter("swe_bench_multilingual"),
    "swe_bench_pro": SweBenchProAdapter(),
    "terminal_bench": TerminalBenchAdapter(),
    "tau2_telecom": Tau2Adapter("tau2_telecom", "telecom"),
    "tau2_retail": Tau2Adapter("tau2_retail", "retail"),
    "tau2_airline": Tau2Adapter("tau2_airline", "airline"),
    "osworld_verified": OSWorldAdapter(),
    "toolathlon": ToolathlonAdapter(),
}


def get_adapter(dataset):
    try:
        return ADAPTERS[dataset]
    except KeyError:
        raise KeyError(
            f"unknown dataset {dataset!r}; available: {', '.join(sorted(ADAPTERS))}")
