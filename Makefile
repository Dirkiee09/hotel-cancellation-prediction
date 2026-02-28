ifeq ($(OS),Windows_NT)
    PYTHON := .venv/Scripts/python.exe
else
    PYTHON := .venv/bin/python
endif

.PHONY: install-dev lint format typecheck test security deps-audit train eval benchmark repro-check artifact-check metrics-gate thesis-analysis thesis-analysis-fast

install-dev:
	$(PYTHON) -m pip install -r requirements.txt

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format --check .

typecheck:
	$(PYTHON) -m mypy

test:
	$(PYTHON) -m pytest

security:
	$(PYTHON) -m bandit -q -r src scripts -s B101 -x tests

deps-audit:
	$(PYTHON) -m pip_audit -r requirements.txt --no-deps --disable-pip

train:
	$(PYTHON) scripts/train.py

eval:
	$(PYTHON) scripts/verify.py

benchmark:
	$(PYTHON) scripts/benchmark.py

repro-check:
	$(PYTHON) scripts/repro.py --max-rows 5000

artifact-check:
	$(PYTHON) scripts/check_artifacts.py

metrics-gate:
	$(PYTHON) scripts/check_metrics.py

thesis-analysis: train
	$(PYTHON) scripts/thesis.py

thesis-analysis-fast: train
	$(PYTHON) scripts/thesis.py --skip-tuning --skip-shap
