"""Clear, execute, and validate cancellation notebooks deterministically."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
NOTEBOOKS = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))


def _load_nb(path: Path):
    import nbformat

    return nbformat.read(path, as_version=4)


def _save_nb(path: Path, nb) -> None:
    import nbformat

    with path.open("w", encoding="utf-8", newline="\n") as handle:
        nbformat.write(nb, handle)


def clear_outputs(path: Path) -> None:
    nb = _load_nb(path)
    for cell in nb.cells:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    _save_nb(path, nb)


def execute_notebook(path: Path, timeout_sec: int) -> None:
    from nbconvert.preprocessors import ExecutePreprocessor

    nb = _load_nb(path)
    ep = ExecutePreprocessor(timeout=timeout_sec, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(path.parent)}})
    _save_nb(path, nb)


def check_execution_consistency(path: Path) -> None:
    nb = _load_nb(path)

    def _source_text(cell: dict) -> str:
        source = cell.get("source", "")
        if isinstance(source, list):
            return "".join(source)
        return str(source)

    counts = [
        cell.get("execution_count")
        for cell in nb.cells
        if cell.get("cell_type") == "code" and _source_text(cell).strip()
    ]
    if not counts:
        raise RuntimeError(f"{path}: no executable code cells found")
    if any(count is None for count in counts):
        raise RuntimeError(f"{path}: found code cells without execution_count")
    if len(set(counts)) != len(counts):
        raise RuntimeError(f"{path}: duplicate execution_count values detected")
    if counts != sorted(counts):
        raise RuntimeError(f"{path}: non-monotonic execution_count values detected")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deterministic clear+run+check for cancellation notebooks."
    )
    parser.add_argument(
        "--clear-only",
        action="store_true",
        help="Only clear outputs/execution counts, do not execute.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=5400,
        help="Per-notebook execution timeout in seconds (default: 5400).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTHONHASHSEED", "42")
    if os.name == "nt":
        # Windows ACL writes can fail in constrained environments; allow insecure fallback.
        os.environ.setdefault("JUPYTER_ALLOW_INSECURE_WRITES", "true")
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    if existing_pythonpath:
        os.environ["PYTHONPATH"] = f"{PROJECT_ROOT}{os.pathsep}{existing_pythonpath}"
    else:
        os.environ["PYTHONPATH"] = str(PROJECT_ROOT)

    for nb_path in NOTEBOOKS:
        if not nb_path.exists():
            raise FileNotFoundError(f"Notebook not found: {nb_path}")

        print(f"[step] clear outputs: {nb_path}")
        clear_outputs(nb_path)

        if args.clear_only:
            continue

        print(f"[step] execute: {nb_path}")
        execute_notebook(nb_path, timeout_sec=args.timeout_sec)

        print(f"[step] consistency check: {nb_path}")
        check_execution_consistency(nb_path)

    if args.clear_only:
        print("[done] notebooks cleared (no execution).")
    else:
        print("[done] notebooks executed and validated.")


if __name__ == "__main__":
    main()
