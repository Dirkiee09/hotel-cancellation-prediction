"""Run the end-to-end training pipeline with optional post-train steps.

Usage examples:
    python scripts/train.py                          # train only
    python scripts/train.py --verify                 # train + verification report
    python scripts/train.py --thesis                 # train + thesis analysis
    python scripts/train.py --thesis --skip-shap     # train + fast thesis
    python scripts/train.py --verify-only            # verify existing artifacts (no train)
    python scripts/train.py --repro --max-rows 5000  # reproducibility check (no train)
"""

from __future__ import annotations

import argparse
import logging

from src.pipelines import run_training_pipeline
from src.utils import configure_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the hotel booking cancellation model end-to-end.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to the hotel bookings CSV. Defaults to data/hotel_bookings.csv.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        metavar="N",
        help="Limit training to the first N rows (useful for fast smoke-tests).",
    )

    # Post-train steps
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run model verification after training.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Run model verification on existing artifacts (skip training).",
    )
    parser.add_argument(
        "--thesis",
        action="store_true",
        help="Run thesis-grade analysis after training.",
    )
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Skip Optuna hyperparameter tuning (only with --thesis).",
    )
    parser.add_argument(
        "--skip-shap",
        action="store_true",
        help="Skip SHAP feature importance analysis (only with --thesis).",
    )
    parser.add_argument(
        "--repro",
        action="store_true",
        help="Run reproducibility check instead of training.",
    )
    args = parser.parse_args()

    configure_logging()

    # ── Reproducibility check (standalone, no training) ──────────────
    if args.repro:
        from src.eval.repro import main as repro_main

        repro_main()
        return

    # ── Training ─────────────────────────────────────────────────────
    if not args.verify_only:
        outputs = run_training_pipeline(data_path=args.data_path, max_rows=args.max_rows)
        logger.info(
            "pipeline_finished model_path=%s metrics_file=%s",
            outputs.model_path,
            outputs.reports_dir / "metrics.json",
        )

    # ── Verification ─────────────────────────────────────────────────
    if args.verify or args.verify_only:
        from src.eval import run_model_verification

        v_out = run_model_verification()
        logger.info("verification_report_generated path=%s", v_out["report_path"])
        logger.info("verification_figures_dir path=%s", v_out["figures_dir"])

    # ── Thesis analysis ──────────────────────────────────────────────
    if args.thesis:
        from src.eval.thesis import run_thesis_analysis

        t_out = run_thesis_analysis(
            skip_tuning=args.skip_tuning,
            skip_shap=args.skip_shap,
            max_rows=args.max_rows,
        )
        logger.info("thesis_analysis_finished reports_dir=%s", t_out.reports_dir)
        for name, path in t_out.sections.items():
            logger.info("  section=%s path=%s", name, path)


if __name__ == "__main__":
    main()
