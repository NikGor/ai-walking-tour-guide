FROM python:3.11-slim

WORKDIR /app

# Install poetry (pinned version for reproducibility)
RUN pip install --no-cache-dir poetry==1.8.3

# Copy dependency files first — layer cache stays valid unless deps change
COPY pyproject.toml poetry.lock* ./

# Install runtime deps only, no venv (we're inside a container)
RUN poetry config virtualenvs.create false \
    && poetry install --no-root --without dev --no-interaction --no-ansi

# Copy source
COPY . .

# ── Database location ──────────────────────────────────────────────────────
# Local: DB lives next to the project, shared with the host via volume mount
ENV SOLARIS_DB_URL=sqlite+aiosqlite:///./data/solaris.db
# Prod: uncomment the line below and comment out the one above
# ENV SOLARIS_DB_URL=sqlite+aiosqlite:////var/lib/solaris/solaris.db
# ──────────────────────────────────────────────────────────────────────────

RUN mkdir -p data
VOLUME /app/data

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
