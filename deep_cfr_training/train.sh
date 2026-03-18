#!/bin/bash
# Deep CFR Training Script
#
# RunPod H200 (139GB) + 192vCPU (8h target):
#   bash train.sh 1200 5000 131072 50 50 1 2000000
# RunPod H100 SXM5 (80GB) + 16+ vCPU:
#   bash train.sh 800 5000 65536 50 50 1 1000000
# Local macOS (MPS):
#   bash train.sh 50 200 4096 5 20 1 500000
set -e

ITERS=${1:-1200}
TRAVERSALS=${2:-5000}
BATCH_SIZE=${3:-131072}
TRAIN_BATCHES=${4:-50}
DISC_GAMES=${5:-50}
N_TRAV_THREADS=${6:-1}
BUFFER_SIZE=${7:-2000000}


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

python trainer.py \
    --iterations      "$ITERS"          \
    --traversals      "$TRAVERSALS"     \
    --batch-size      "$BATCH_SIZE"     \
    --train-batches   "$TRAIN_BATCHES"  \
    --discard-n-games "$DISC_GAMES"     \
    --n-trav-threads  "$N_TRAV_THREADS" \
    --buffer-size     "$BUFFER_SIZE"    \
    --checkpoint-every 100              \
    --output model/deep_cfr

echo ""
echo "=== Training Complete ==="
ls -lh model/deep_cfr*.pt model/deep_cfr*.pkl 2>/dev/null || true
