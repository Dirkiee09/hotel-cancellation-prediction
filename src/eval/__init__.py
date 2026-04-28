"""Evaluation utilities for reproducible model verification.

Uses lazy imports so that lightweight scripts (e.g. verify.py) don't pay the
startup cost of importing heavy modules (benchmark, thesis) that pull in
SHAP, Optuna, etc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .benchmark import run_model_benchmark as run_model_benchmark
    from .repro import run_repro_check as run_repro_check
    from .thesis import run_thesis_analysis as run_thesis_analysis
    from .verify import run_model_verification as run_model_verification

__all__ = [
    "run_model_verification",
    "run_repro_check",
    "run_thesis_analysis",
    "run_model_benchmark",
]


def __getattr__(name: str) -> object:
    if name == "run_model_verification":
        from .verify import run_model_verification

        return run_model_verification
    if name == "run_model_benchmark":
        from .benchmark import run_model_benchmark

        return run_model_benchmark
    if name == "run_thesis_analysis":
        from .thesis import run_thesis_analysis

        return run_thesis_analysis
    if name == "run_repro_check":
        from .repro import run_repro_check

        return run_repro_check
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
