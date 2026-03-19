#!/usr/bin/env bash
# train.sh — Build C++ agent and run training pipeline.
#
# Usage:
#   ./train.sh                          # compile + train all phases
#   ./train.sh preflop                  # compile + train preflop only
#   ./train.sh discard                  # compile + train discard only
#   ./train.sh --no-compile preflop     # skip compilation, train preflop
#
# Options (set via env vars):
#   PREFLOP_ITERS=200000   MCCFR iterations for preflop
#   DISCARD_HANDS=5000     hands for discard training
#   DISCARD_EPOCHS=100     epochs for DiscardNet
#   DISCARD_SIMS=50        MC sims per EV estimate

set -e
cd "$(dirname "$0")"

# ── Defaults ──────────────────────────────────────────────────────────────────
PHASE="${1:-all}"
COMPILE=true
if [ "$1" = "--no-compile" ]; then
    COMPILE=false
    PHASE="${2:-all}"
fi

PREFLOP_ITERS="${PREFLOP_ITERS:-200000}"
DISCARD_HANDS="${DISCARD_HANDS:-5000}"
DISCARD_EPOCHS="${DISCARD_EPOCHS:-100}"
DISCARD_SIMS="${DISCARD_SIMS:-50}"
N_WORKERS="${N_WORKERS:-$(python3 -c 'import os; print(os.cpu_count())')}"

# ── Compile C++ prob agent ─────────────────────────────────────────────────────
if [ "$COMPILE" = true ]; then
    echo "==> Compiling cpp/prob_agent.cpp ..."
    if command -v clang++ &>/dev/null; then
        CXX=clang++
    elif command -v g++ &>/dev/null; then
        CXX=g++
    else
        echo "ERROR: no C++ compiler found (clang++ or g++ required)" >&2
        exit 1
    fi
    echo "    Using compiler: $CXX"
    $CXX -O3 -std=c++17 -shared -fPIC \
        -o cpp/prob_agent.so \
        cpp/prob_agent.cpp
    echo "    Built: cpp/prob_agent.so"
fi

# ── Run training ───────────────────────────────────────────────────────────────
echo "==> Starting training: phase=$PHASE"
echo "    PREFLOP_ITERS=$PREFLOP_ITERS  N_WORKERS=$N_WORKERS"
echo "    DISCARD_HANDS=$DISCARD_HANDS  DISCARD_EPOCHS=$DISCARD_EPOCHS  DISCARD_SIMS=$DISCARD_SIMS"
echo ""

python train.py "$PHASE" \
    --iters   "$PREFLOP_ITERS" \
    --hands   "$DISCARD_HANDS" \
    --epochs  "$DISCARD_EPOCHS" \
    --sims    "$DISCARD_SIMS" \
    --workers "$N_WORKERS"

echo ""
echo "==> Done. Models saved to models/"
