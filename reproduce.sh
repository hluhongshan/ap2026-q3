#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  export PATH="$ROOT/.venv/bin:$PATH"
fi
export TORCH_HOME="$ROOT/.cache/torch"
export XDG_CACHE_HOME="$ROOT/.cache"

QUICK=0
case "${1:-}" in
  --quick) QUICK=1 ;;
  "") ;;
  *) echo "usage: $0 [--quick]" >&2; exit 2 ;;
esac

run_step() {
  local name="$1"
  shift
  echo
  echo "=== ${name} ==="
  local start
  start="$(date +%s)"
  if "$@"; then
    local elapsed=$(( $(date +%s) - start ))
    echo "=== ${name}: ${elapsed}s ==="
  else
    local status=$?
    local elapsed=$(( $(date +%s) - start ))
    echo "=== ${name}: FAILED after ${elapsed}s (exit ${status}) ===" >&2
    exit "$status"
  fi
}

if [[ "$QUICK" -eq 0 ]]; then
  run_step "data fetch offline" python data/fetch.py --offline
fi

run_step "tests" python -m unittest discover -s tests -v

if [[ "$QUICK" -eq 0 ]]; then
  run_step "approach1 Q3A classical" python code/approach1_q3a_classical.py
  run_step "approach1 Q3A post-hoc" python code/approach1_q3a_posthoc.py
  run_step "approach2 Q3B sequence" python code/approach2_q3b_sequence.py
  run_step "approach2 Q3B post-hoc" python code/approach2_q3b_posthoc.py --mode all
fi

run_step "compare" python code/compare.py
