.PHONY: install install-ml test lint fmt

# Core + dev tools. Never touches torch (see README: "Why ML libraries are optional").
install:
	pip install -e ".[dev]"

# Only on a machine WITHOUT a working torch. Never run this on SageMaker.
install-ml:
	pip install -e ".[ml,dev]"

test:
	python -m pytest -q

lint:
	ruff check .
	ruff format --check .

fmt:
	ruff format .
	ruff check --fix .
