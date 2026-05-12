"""PostToolUse hook: auto-format Python files after Claude edits them.

Reads the event JSON from stdin, extracts the modified file path, and runs
`ruff format` on it if it's a .py file. Silent on success; prints a one-liner
on failure so Claude sees it in the next turn.

Exit code is always 0 — formatting issues should be informational, not blocking.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0

    file_path = event.get("tool_input", {}).get("file_path", "")
    if not file_path or not file_path.endswith(".py"):
        return 0

    if not Path(file_path).exists():
        return 0

    try:
        result = subprocess.run(  # noqa: S603
            ["ruff", "format", file_path],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 0

    if result.returncode != 0:
        sys.stderr.write(f"[hook] ruff format failed on {file_path}\n")
        sys.stderr.write(result.stderr[-500:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
