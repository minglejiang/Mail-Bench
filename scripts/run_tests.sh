#!/usr/bin/env bash
# Run the MAIL-Bench unit test suite (no network, no simulator).
# Usage: scripts/run_tests.sh [extra pytest args]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python3 -c "import pytest" 2>/dev/null || { echo "pytest not found: pip install -e '.[test]'" >&2; exit 2; }
exec python3 -m pytest "$@"
