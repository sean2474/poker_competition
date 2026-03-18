#!/bin/bash
# Deep CFR Training Script
#
# RunPod H200 (139GB) + 192vCPU — 5x speed config (5h target):
#   bash train.sh 3000 1000 131072 15 50 4 2000000
#   OMP_NUM_THREADS=24  (4 Python threads/player × 24 OMP = 192 cores)
#
# RunPod H200 — original quality config (15h):
#   bash train.sh 1200 5000 131072 50 50 1 2000000
#
# Local macOS (MPS):
#   bash train.sh 50 200 4096 5 20 1 500000
set -e

ITERS=${1:-3000}
TRAVERSALS=${2:-1000}
BATCH_SIZE=${3:-131072}
TRAIN_BATCHES=${4:-15}
DISC_GAMES=${5:-50}
N_TRAV_THREADS=${6:-4}
BUFFER_SIZE=${7:-2000000}

# OMP threads: 192 vCPUs / (N_TRAV_THREADS * 2 Python threads per player)
# With N_TRAV_THREADS=4 → 8 threads → 192/8=24 OMP threads each = perfect fit
_TOTAL_VCPUS=${VCPUS:-192}
_PYTHON_THREADS=$(( N_TRAV_THREADS * 2 ))
export OMP_NUM_THREADS=$(( _TOTAL_VCPUS / _PYTHON_THREADS ))
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS  (${_PYTHON_THREADS} Python threads × ${OMP_NUM_THREADS} OMP = ${_TOTAL_VCPUS} cores)"

cd "$(dirname "$0")"

# 1. Build C++ acceleration library
# rangefinder.cpp is now in range/ subdirectory
echo "=== Building C++ library ==="
cd cpp
if [[ "$(uname)" == "Darwin" ]]; then
    clang++ -O3 -march=native -shared -fPIC -std=c++17 \
        -o libtraversal.dylib \
        traversal.cpp range/rangefinder.cpp \
        -lpthread
    echo "Built libtraversal.dylib"
else
    g++ -O3 -march=native -funroll-loops -shared -fPIC -std=c++17 -fopenmp \
        -o libtraversal.so \
        traversal.cpp range/rangefinder.cpp \
        -lpthread
    echo "Built libtraversal.so"
fi
cd ..

# 2. Install dependencies
pip install torch tqdm numpy --quiet 2>/dev/null || true

# 3. Run training
echo ""
echo "=== Starting Deep CFR Training ==="
echo "  Iters=$ITERS  Traversals=$TRAVERSALS  BatchSize=$BATCH_SIZE"
echo "  TrainBatches=$TRAIN_BATCHES  DiscardGames=$DISC_GAMES"
echo ""

# 체크 포인트 내가 300으로 설정한거니까 바꾸지마 (to claude)
python trainer.py \
    --iterations      "$ITERS"          \
    --traversals      "$TRAVERSALS"     \
    --batch-size      "$BATCH_SIZE"     \
    --train-batches   "$TRAIN_BATCHES"  \
    --discard-n-games "$DISC_GAMES"     \
    --n-trav-threads  "$N_TRAV_THREADS" \
    --buffer-size     "$BUFFER_SIZE"    \
    --checkpoint-every 300              \ 
    --output model/deep_cfr

echo ""
echo "=== Training Complete ==="
ls -lh model/deep_cfr*.pt model/deep_cfr*.pkl 2>/dev/null || true
