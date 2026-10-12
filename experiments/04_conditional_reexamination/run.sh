#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PLSAR_ROOT="${PLSAR_ROOT:-$ROOT}"
mode="${1:-final}"
shift || true
case "$mode" in
  final) exec python "$ROOT/src/final_ast_complementary_paper_eval.py" "$@" ;;
  cascade) exec python "$ROOT/src/fisher_retaining_cascade.py" "$@" --root "$PLSAR_ROOT" ;;
  *) echo "unknown mode: $mode" >&2; exit 2 ;;
esac
