.PHONY: run stop

run:
	poetry run uvicorn main:app --host 0.0.0.0 --port 8000 --reload

stop:
	fuser -k 8000/tcp 2>/dev/null || true
