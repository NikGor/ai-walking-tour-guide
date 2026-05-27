.PHONY: run stop lint fmt check install-hooks

run:
	poetry run uvicorn main:app --host 0.0.0.0 --port 8000 --reload

stop:
	fuser -k 8000/tcp 2>/dev/null || true

lint:
	poetry run ruff check .

fmt:
	poetry run ruff format .

check: lint
	poetry run pytest

install-hooks:
	poetry run pre-commit install
