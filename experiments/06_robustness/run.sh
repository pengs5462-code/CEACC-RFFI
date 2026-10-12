#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PLSAR_ROOT="${PLSAR_ROOT:-$ROOT}"
mode="${1:?usage: run.sh mode [arguments...]}"
shift
case "$mode" in
  temporal) exec python "$ROOT/src/evaluate_final_ast_temporal_shift.py" "$@" ;;
  openness) exec python "$ROOT/src/aggregate_openness_robustness.py" "$@" ;;
  *) echo "unknown robustness study: $mode" >&2; exit 2 ;;
esac
