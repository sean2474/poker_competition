#!/bin/bash
# Convert C++ binary strategy and copy to submission/data for deployment.
# Usage: bash training/deploy.sh

DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/.." && pwd)"

cd "$ROOT"
source .venv/bin/activate 2>/dev/null

# Convert binary to numpy
python "$DIR/cpp/convert_to_python.py" "$DIR/data/strategy_cpp.bin" "$ROOT/submission/data"

echo ""
echo "Deployed to submission/data/"
ls -lh submission/data/*.npy submission/data/*.pkl 2>/dev/null
