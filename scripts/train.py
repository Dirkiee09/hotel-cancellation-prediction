"""Run the end-to-end training pipeline."""

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
    args = parser.parse_args()

    configure_logging()
    outputs = run_training_pipeline(data_path=args.data_path, max_rows=args.max_rows)
    logger.info(
        "pipeline_finished model_path=%s metrics_file=%s",
        outputs.model_path,
        outputs.reports_dir / "metrics.json",
    )


if __name__ == "__main__":
    main()
