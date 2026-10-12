#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PLSAR_ROOT="${PLSAR_ROOT:-$ROOT}"
exec python "$ROOT/src/final_ast_complementary_paper_eval.py" "$@"
