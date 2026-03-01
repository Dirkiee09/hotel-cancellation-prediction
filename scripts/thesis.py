"""Run thesis-grade analysis suite."""

from __future__ import annotations

import argparse
import logging

from src.eval.thesis import run_thesis_analysis
from src.utils import configure_logging

logger = logging.getLogger(__name__)


def main() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(description="Run thesis-grade model analyses.")
    parser.add_argument(
        "--skip-tuning", action="store_true", help="Skip Optuna hyperparameter tuning"
    )
    parser.add_argument(
        "--skip-shap", action="store_true", help="Skip SHAP feature importance analysis"
    )
    parser.add_argument(
        "--max-rows", type=int, default=None, help="Limit rows for faster iteration"
    )
    args = parser.parse_args()

    outputs = run_thesis_analysis(
        skip_tuning=args.skip_tuning,
        skip_shap=args.skip_shap,
        max_rows=args.max_rows,
    )

    logger.info("thesis_analysis_finished reports_dir=%s", outputs.reports_dir)
    for name, path in outputs.sections.items():
        logger.info("  section=%s path=%s", name, path)


if __name__ == "__main__":
    main()
