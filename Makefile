.PHONY: setup lock test test-python test-contract lint format preflight compile demo

setup:
	uv sync --all-extras --group dev
	npm ci

lock:
	uv lock
	npm install --package-lock-only

test: test-python test-contract

test-python:
	uv run --group dev pytest --cov=sigil --cov-report=term-missing

test-contract:
	npm test

lint:
	uv run --group dev ruff check .
	uv run --group dev ruff format --check .
	uv run --group dev mypy sigil

format:
	uv run --group dev ruff check --fix .
	uv run --group dev ruff format .

preflight:
	uv run sigil preflight

compile:
	npm run compile

demo:
	uv run sigil run --help
