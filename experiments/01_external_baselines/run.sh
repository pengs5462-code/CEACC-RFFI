#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$ROOT/src:$ROOT/baselines${PYTHONPATH:+:$PYTHONPATH}"
method="${1:?usage: run.sh baseline [arguments...]}"
shift
case "$method" in
  hyperrsi) exec python "$ROOT/baselines/train_hyperrsi_official.py" "$@" ;;
  nsrff) exec python "$ROOT/baselines/train_nsrff_official.py" "$@" ;;
  openmax) exec python "$ROOT/baselines/evaluate_openmax_multiknown17.py" "$@" ;;
  *) echo "unknown baseline: $method" >&2; exit 2 ;;
esac
