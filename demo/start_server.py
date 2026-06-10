"""Start the FastAPI + Gradio server for demo.

Usage:
    python demo/start_server.py              # default: localhost:8000
    python demo/start_server.py --port 9000  # custom port
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import webbrowser
from pathlib import Path
from time import sleep

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the prediction server.")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser")
    args = parser.parse_args()

    # Verify artifacts exist before starting
    model_path = ROOT / "artifacts" / "best_model.pkl"
    if not model_path.exists():
        print("ERROR: No trained model found.")
        print("Run this first:  python scripts/train.py")
        sys.exit(1)

    url = f"http://localhost:{args.port}"
    print(f"Starting server at {url}")
    print(f"  API docs:  {url}/docs")
    print(f"  Gradio UI: {url}/ui")
    print(f"  Health:    {url}/healthz")
    print()
    print("Press Ctrl+C to stop.\n")

    proc = subprocess.Popen(  # noqa: S603
        [
            sys.executable,
            "-m",
            "uvicorn",
            "src.app.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(args.port),
        ],
        cwd=str(ROOT),
    )

    if not args.no_browser:
        sleep(2)
        webbrowser.open(f"{url}/ui")

    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
