"""Deterministic execution helpers."""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Set process-level deterministic seeds for supported libraries.

    Note: PYTHONHASHSEED only takes effect when set before interpreter startup
    (e.g. via environment variable). Setting it here is a no-op for hash
    randomization but is kept for documentation purposes.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
