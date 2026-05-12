"""PreToolUse hook: refuse edits to canonical inputs and generated artifacts.

Reads the event JSON from stdin and blocks Edit/Write attempts on:
    - data/hotel_bookings.csv      (immutable input)
    - artifacts/*.pkl              (training outputs)
    - artifacts/*.keras            (ADR neural net)
    - artifacts/*.json             (feature/threshold/metadata schemas)

Exit code 2 is the conventional "block with reason" signal — Claude Code shows
the stderr message to the model so it can self-correct.

This makes the CLAUDE.md "Never do" list self-enforcing instead of relying on
the model to remember every rule.
"""

from __future__ import annotations

import fnmatch
import json
import sys
from pathlib import PurePosixPath

BLOCKED_PATTERNS: tuple[str, ...] = (
    "data/hotel_bookings.csv",
    "artifacts/*.pkl",
    "artifacts/*.keras",
    "artifacts/*.json",
)

REMEDIATION = (
    "Blocked: canonical input or generated artifact. "
    "Regenerate via `python scripts/train.py`, do not hand-edit. "
    "If you really need to edit this, ask the user first."
)


def _normalize(path: str) -> str:
    return str(PurePosixPath(path.replace("\\", "/")))


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0

    file_path = event.get("tool_input", {}).get("file_path", "")
    if not file_path:
        return 0

    rel = _normalize(file_path)
    for pattern in BLOCKED_PATTERNS:
        if fnmatch.fnmatch(rel, f"*{pattern}") or fnmatch.fnmatch(rel, pattern):
            sys.stderr.write(f"{REMEDIATION}\nPath: {file_path}\nPattern: {pattern}\n")
            return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
