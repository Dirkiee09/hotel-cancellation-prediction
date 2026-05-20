"""Start the FastAPI + Gradio server for the PH (Philippine dataset) sub-study.

Parallel to ``demo/start_server.py`` but targets the Philippine dataset
sub-study on a different port (8001 by default) with its own readiness
checks. Both servers can run simultaneously for a side-by-side defense
demo without conflict.

Bullet-proof launcher that:
  1. Verifies trained PH artifacts exist before spawning uvicorn.
  2. Refuses to start if port 8001 is already taken.
  3. Polls ``/healthz`` and only opens the browser once the model is loaded.
  4. Tails the PH uvicorn log if the process exits early.

Usage:
    python demo/start_server_ph.py                # http://localhost:8001/ui
    python demo/start_server_ph.py --port 9001    # custom port
    python demo/start_server_ph.py --no-browser   # CI / headless
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / ".gradio"
LOG_PATH = LOG_DIR / "uvicorn_ph.log"

HEALTHZ_TIMEOUT_SEC = 30
HEALTHZ_POLL_INTERVAL = 0.5


def _is_port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        try:
            sock.connect((host, port))
            return True
        except (ConnectionRefusedError, TimeoutError, OSError):
            return False


def _wait_for_healthz(url: str, proc: subprocess.Popen[bytes]) -> bool:
    """Poll until ``/healthz`` returns 200 or the process exits."""
    deadline = time.time() + HEALTHZ_TIMEOUT_SEC
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"{url}/healthz", timeout=1) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionRefusedError, TimeoutError, OSError):
            pass
        time.sleep(HEALTHZ_POLL_INTERVAL)
    return False


def _tail_log(path: Path, lines: int = 30) -> str:
    if not path.exists():
        return "(no log captured)"
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(text[-lines:])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch the PH (Philippine dataset) prediction server."
    )
    parser.add_argument("--port", type=int, default=8001, help="Port to serve on (default: 8001)")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address")
    parser.add_argument("--no-browser", action="store_true", help="Skip opening browser")
    args = parser.parse_args()

    # 1. Artifacts present?
    model_path = ROOT / "artifacts" / "ph" / "ph_model.pkl"
    if not model_path.exists():
        print("ERROR: No trained PH model found.")
        print(f"  Expected: {model_path}")
        print("  Fix:      python scripts/train_ph.py")
        return 1

    # 2. Port free?
    if _is_port_in_use("127.0.0.1", args.port):
        print(f"ERROR: Port {args.port} is already in use.")
        print(f"  Check existing server: curl http://localhost:{args.port}/healthz")
        print("  Stop running uvicorn (Windows PowerShell):")
        print("      Get-CimInstance Win32_Process | Where-Object {")
        print("        $_.CommandLine -match 'ph_main' } | Stop-Process -Force")
        print("  Or pick a different port: python demo/start_server_ph.py --port 9001")
        return 1

    url = f"http://localhost:{args.port}"
    print(f"Starting PH sub-study server at {url}")
    print(f"  API docs:  {url}/docs")
    print(f"  Gradio UI: {url}/ui")
    print(f"  Health:    {url}/healthz")
    print(f"  Log file:  {LOG_PATH}")
    print()
    print("Note: predictions are logged to data/predictions/ph_predictions.sqlite")
    print("      and auto-exported to ph_predictions_live.csv for Power BI.")
    print()

    # 3. Spawn uvicorn with output captured for failure diagnostics.
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_PATH.open("w", encoding="utf-8")
    try:
        proc = subprocess.Popen(  # noqa: S603
            [
                sys.executable,
                "-m",
                "uvicorn",
                "src.app.ph_main:app",
                "--host",
                args.host,
                "--port",
                str(args.port),
            ],
            cwd=str(ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    except OSError as exc:
        print(f"ERROR: Failed to spawn uvicorn: {exc}")
        log_file.close()
        return 1

    # 4. Wait for readiness before opening browser.
    print("Waiting for PH server to become ready...")
    ready = _wait_for_healthz(url, proc)
    if ready:
        print("PH server ready.")
        if not args.no_browser:
            webbrowser.open(f"{url}/ui")
        print("\nPress Ctrl+C to stop.")
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            print("\nPH server stopped.")
        finally:
            log_file.close()
        return 0

    # Not ready — figure out why and surface it.
    log_file.close()
    if proc.poll() is not None:
        print(f"\nERROR: uvicorn exited with code {proc.returncode} before becoming ready.")
    else:
        print(f"\nERROR: /healthz did not respond within {HEALTHZ_TIMEOUT_SEC}s.")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    print(f"\nLast lines of {LOG_PATH}:")
    print(_tail_log(LOG_PATH))
    return 1


if __name__ == "__main__":
    sys.exit(main())
