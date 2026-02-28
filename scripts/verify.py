"""Run deterministic model verification report generation."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval import run_model_verification
from src.utils import configure_logging

logger = logging.getLogger(__name__)


def main() -> None:
    configure_logging()
    outputs = run_model_verification()
    logger.info("verification_report_generated path=%s", outputs["report_path"])
    logger.info("verification_figures_dir path=%s", outputs["figures_dir"])


if __name__ == "__main__":
    main()
