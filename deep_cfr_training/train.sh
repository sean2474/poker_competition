#!/bin/bash
# Deep CFR Training Script
# Usage: bash train.sh [iterations] [traversals]
#
# RunPod: bash train.sh 500 2000
# Local:  bash train.sh 50 200
set -e

ITERS=${1:-500}
TRAVERSALS=${2:-1000}
BATCH_SIZE=${3:-32768}   # optimal: MPS peaks here (1046K/s), CPU plateaus (520K/s)
TRAIN_BATCHES=${4:-100}  # 32768x100 = 3.3M samples per iter

cd "$(dirname "$0")"

# 1. Build C++ acceleration library
echo "=== Building C++ library ==="
cd cpp
if [[ "$(uname)" == "Darwin" ]]; then
    g++ -O3 -shared -fPIC -std=c++17 -o libtraversal.dylib traversal.cpp -lpthread
    echo "Built libtraversal.dylib"
else
    g++ -O3 -shared -fPIC -std=c++17 -o libtraversal.so traversal.cpp -lpthread -fopenmp
    echo "Built libtraversal.so"
fi
cd ..

# 2. Install dependencies
echo ""
echo "=== Installing dependencies ==="
pip install torch tqdm numpy 2>/dev/null || true

# 3. Run training
echo ""
echo "=== Starting Deep CFR Training ==="
echo "Iterations: $ITERS, Traversals: $TRAVERSALS"
echo "Batch size: $BATCH_SIZE, Train batches: $TRAIN_BATCHES"
echo ""

python deep_cfr.py \
    --iterations "$ITERS" \
    --traversals "$TRAVERSALS" \
    --batch-size "$BATCH_SIZE" \
    --train-batches "$TRAIN_BATCHES"

echo ""
echo "=== Training Complete ==="
echo "Model saved to: model/deep_cfr_strategy.pt"
ls -lh model/deep_cfr*.pt 2>/dev/null || true
