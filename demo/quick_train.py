"""Quick smoke-train for demo prep (10k rows, ~30 seconds).

Use this when you need artifacts fast and don't care about full accuracy.
For the real model, run: python scripts/train.py

Usage:
    python demo/quick_train.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    model_path = ROOT / "artifacts" / "best_model.pkl"

    if model_path.exists():
        print(f"Model already exists: {model_path}")
        answer = input("Retrain with 10k rows? [y/N] ").strip().lower()
        if answer != "y":
            print("Skipped. Start the server with: python demo/start_server.py")
            return

    print("Training on 10k rows (fast smoke-train)...")
    print("For full accuracy, run: python scripts/train.py\n")

    result = subprocess.run(  # noqa: S603
        [sys.executable, "scripts/train.py", "--max-rows", "10000"],
        cwd=str(ROOT),
    )

    if result.returncode == 0:
        print("\nTraining complete. Start the server with:")
        print("  python demo/start_server.py")
    else:
        print("\nTraining failed. Check the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
