"""Run the end-to-end training pipeline."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines import run_training_pipeline
from src.utils import configure_logging

logger = logging.getLogger(__name__)


def main() -> None:
    configure_logging()
    outputs = run_training_pipeline()
    logger.info(
        "pipeline_finished model_path=%s metrics_file=%s",
        outputs.model_path,
        outputs.reports_dir / "metrics.json",
    )


if __name__ == "__main__":
    main()
