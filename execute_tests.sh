#!/usr/bin/env bash
set -euo pipefail

# ── execute_tests.sh ──────────────────────────────────────────────────────────
# Run the full test suite.
#
# Usage:
#   ./execute_tests.sh           # unit + integration
#   ./execute_tests.sh unit      # unit tests only (fast, no API calls)
#   ./execute_tests.sh int       # integration tests only (real LLM)
# ─────────────────────────────────────────────────────────────────────────────

MODE="${1:-all}"

case "$MODE" in
  unit)
    echo "▶ Unit tests only"
    poetry run pytest tests/unit/ -v
    ;;
  int|integration)
    echo "▶ Integration tests only (real LLM)"
    poetry run pytest tests/integration/ -v -m integration
    ;;
  all|*)
    echo "▶ Unit tests"
    poetry run pytest tests/unit/ -v

    echo ""
    echo "▶ Integration tests (real LLM)"
    poetry run pytest tests/integration/ -v -m integration
    ;;
esac
