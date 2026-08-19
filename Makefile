.PHONY: install run debug clean lint lint-strict

FUNCTIONS ?= data/input/functions_definition.json
INPUT ?= data/input/function_calling_tests.json
OUTPUT ?= data/output/function_calling_results.json

install:
	uv sync

run:
	uv run python -m src \
		--functions_definition $(FUNCTIONS) \
		--input $(INPUT) \
		--output $(OUTPUT)

debug:
	uv run python -m pdb -m src \
		--functions_definition $(FUNCTIONS) \
		--input $(INPUT) \
		--output $(OUTPUT)

lint:
	uv run flake8 src/
	uv run mypy src/ --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 src/
	uv run mypy src/ --strict

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache data/output