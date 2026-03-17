#!/bin/bash
# Deep CFR Training Script
# Usage: bash train.sh [iterations] [traversals]
#
# RTX 3090 (24GB VRAM) + 32 vCPU + 125GB RAM:
#   bash train.sh 500 2000
# Local (MPS/CPU):
#   bash train.sh 50 200
set -e

ITERS=${1:-500}
TRAVERSALS=${2:-2000}    # 32 vCPU: C++ OpenMP parallelizes batch_deal_discard
BATCH_SIZE=${3:-131072}  # RTX 3090 24GB: 128K optimal (vs 65536 for smaller GPUs)
TRAIN_BATCHES=${4:-100}  # 128K×100 = 12.8M samples/iter (4M buffer → 3× coverage)
BUFFER_SIZE=${5:-4000000} # 4M reservoir (125GB RAM can hold ~8GB of samples easily)

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

python run.py \
    --iterations    "$ITERS"        \
    --traversals    "$TRAVERSALS"   \
    --batch-size    "$BATCH_SIZE"   \
    --train-batches "$TRAIN_BATCHES" \
    --buffer-size   "$BUFFER_SIZE"

echo ""
echo "=== Training Complete ==="
echo "Model saved to: model/deep_cfr_strategy.pt"
ls -lh model/deep_cfr*.pt 2>/dev/null || true
