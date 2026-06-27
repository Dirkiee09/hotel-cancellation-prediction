"""Generate the 16-table model benchmark in reports/benchmarks/.

Thin CLI wrapper over :func:`src.eval.benchmark.run_model_benchmark` so that
``make benchmark`` and CI have a stable, discoverable entry point (the benchmark
logic itself lives in ``src/eval/benchmark.py``).
"""

from __future__ import annotations

from src.eval.benchmark import run_model_benchmark


def main() -> None:
    run_model_benchmark()


if __name__ == "__main__":
    main()
