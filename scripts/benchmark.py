"""Run deterministic multi-model benchmark and export benchmark tables."""

from __future__ import annotations

import argparse
import logging

from src.eval.benchmark import run_model_benchmark
from src.utils import configure_logging

logger = logging.getLogger(__name__)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Run deterministic model benchmark.")
    parser.add_argument(
        "--max-rows", type=int, default=None, help="Optional row cap for faster iteration"
    )
    parser.add_argument(
        "--n-bootstraps",
        type=int,
        default=None,
        help="Override bootstrap iterations for CI/significance tables",
    )
    parser.add_argument(
        "--temporal-buckets",
        type=int,
        default=None,
        help="Override number of temporal stability buckets",
    )
    args = parser.parse_args()

    kwargs = {}
    if args.max_rows is not None:
        kwargs["max_rows"] = args.max_rows
    if args.n_bootstraps is not None:
        kwargs["n_bootstraps"] = args.n_bootstraps
    if args.temporal_buckets is not None:
        kwargs["temporal_buckets"] = args.temporal_buckets

    outputs = run_model_benchmark(**kwargs)
    logger.info(
        "benchmark_finished reports_dir=%s champion_model=%s tables=%d",
        outputs.reports_dir,
        outputs.champion_model,
        len(outputs.table_paths),
    )
    for table_name, path in outputs.table_paths.items():
        logger.info("  table=%s path=%s", table_name, path)


if __name__ == "__main__":
    main()
