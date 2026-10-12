#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
kind="${1:?usage: run.sh study mode [arguments...]}"
shift
case "$kind" in
  fixed-ast) exec python "$ROOT/src/experiment1_fixed_ast.py" "$@" --root "$ROOT" ;;
  complementary) exec python "$ROOT/src/experiment2_complementary_only.py" "$@" --root "$ROOT" ;;
  *) echo "unknown study: $kind" >&2; exit 2 ;;
esac
