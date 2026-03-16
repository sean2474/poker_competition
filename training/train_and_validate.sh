#!/bin/bash
# Train C++ CFR with periodic validation vs ProbabilityAgent
# Usage: bash submission/train_and_validate.sh [iterations] [validate_every]

ITERS=${1:-500000}
VAL_EVERY=${2:-25000}
DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/.." && pwd)"
DATA="$DIR/data"
CPP="$DIR/cpp"

cd "$ROOT"
source .venv/bin/activate 2>/dev/null

# Build
cd "$CPP" && make train_cfr 2>&1 | tail -1
cd "$ROOT"

# Start training with periodic saves
"$CPP/train_cfr" \
    --iterations "$ITERS" \
    --output "$DATA/strategy_cpp.bin" \
    --checkpoint "$DATA/checkpoint_cpp.bin" \
    --validate-every "$VAL_EVERY" &
TRAIN_PID=$!

# Monitor for .ready signal files and run validation
LAST_VALIDATED=0
while kill -0 $TRAIN_PID 2>/dev/null; do
    SIGNAL="$DATA/strategy_cpp.bin.ready"
    if [ -f "$SIGNAL" ]; then
        ITER=$(cat "$SIGNAL")
        if [ "$ITER" != "$LAST_VALIDATED" ]; then
            echo ""
            echo "========================================"
            echo "  VALIDATING at iteration $ITER"
            echo "========================================"
            
            # Convert binary to numpy
            python "$CPP/convert_to_python.py" "$DATA/strategy_cpp.bin" "$DATA" 2>&1 | tail -2
            
            # Run quick match (200 hands)
            python "$DIR/validate_match.py" 2>&1
            
            LAST_VALIDATED="$ITER"
            rm -f "$SIGNAL"
            echo "========================================"
            echo ""
        fi
    fi
    sleep 5
done

# Wait for training to finish
wait $TRAIN_PID

# Final conversion
echo ""
echo "=== FINAL CONVERSION ==="
python "$CPP/convert_to_python.py" "$DATA/strategy_cpp.bin" "$DATA"

# Final validation
echo ""
echo "=== FINAL VALIDATION ==="
python "$DIR/validate_match.py"

echo ""
echo "Training complete!"
