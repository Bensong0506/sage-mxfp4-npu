#!/usr/bin/env bash
# One-shot entrypoint for the FP4 machine: probe -> selftest.
# Usage: bash run.sh            (auto-detect best mode)
#        bash run.sh fp4        (force a mode in selftest)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODE="${1:-auto}"

# --- env (adjust CANN path if different) ---
CANN="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
[ -f "$CANN/../set_env.sh" ] && source "$CANN/../set_env.sh" 2>/dev/null || \
  source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
export LD_LIBRARY_PATH="$CANN/opp/vendors/customize/op_api/lib:${LD_LIBRARY_PATH:-}"

PY="${PYTHON:-python}"
cd "$HERE"

echo "############ STEP 1: probe environment ############"
"$PY" probe_env.py || true

echo
echo "############ STEP 2: self-test (accuracy + speed) ############"
"$PY" selftest.py --mode "$MODE" || true

echo
echo "############ DONE ############"
echo "If FP4/FP8 PASS with speedup>1, real low-bit attention works."
echo "To run the Wan2.1 video demo, see AGENT.md section 4."
