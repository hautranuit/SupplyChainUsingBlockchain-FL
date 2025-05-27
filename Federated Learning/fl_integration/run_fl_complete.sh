#!/bin/bash
# Complete Federated Learning Training Pipeline
# Usage: ./run_fl_complete.sh

echo "🚀 Starting Federated Learning Training Pipeline..."

# Check if we're in the right directory
if [ ! -f "run_federated_learning.py" ]; then
    echo "❌ Error: run_federated_learning.py not found. Please run from fl_integration directory."
    exit 1
fi

# Check if TFF environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: Virtual environment not detected. Attempting to activate..."
    if [ -f "../tff_wsl_env/bin/activate" ]; then
        source "../tff_wsl_env/bin/activate"
        echo "✅ TFF environment activated"
    else
        echo "❌ Error: TFF environment not found. Please check ../tff_wsl_env/"
        exit 1
    fi
fi

# Verify TFF installation
python -c "import tensorflow_federated as tff; print('✅ TFF version:', tff.__version__)" || {
    echo "❌ Error: TensorFlow Federated not properly installed"
    exit 1
}

# Create timestamped output directory
OUTPUT_DIR="results/fl_training_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Check if demo_context.json exists
DATA_SOURCE="../../SupplyChain_dapp/scripts/lifecycle_demo/demo_context.json"
if [ ! -f "$DATA_SOURCE" ]; then
    echo "❌ Error: Data source not found: $DATA_SOURCE"
    echo "Please ensure the Sybil attack simulation (script 07) has been run first."
    exit 1
fi

echo ""
echo "📊 Training Configuration:"
echo "  - Mode: full (all FL modules)"
echo "  - Data Source: $DATA_SOURCE"  
echo "  - Training Rounds: 35 (optimized for attack detection)"
echo "  - Federated Clients: 7"
echo "  - Batch Size: 16"
echo "  - Learning Rate: 0.005"
echo "  - Output Directory: $OUTPUT_DIR"
echo ""

# Start training with comprehensive logging
echo "🎯 Starting FL training..."
python run_federated_learning.py \
  --mode full \
  --input-data-file "$DATA_SOURCE" \
  --num-rounds 35 \
  --num-clients 7 \
  --batch-size 16 \
  --learning-rate 0.005 \
  --verbose \
  --output-dir "$OUTPUT_DIR" \
  2>&1 | tee "$OUTPUT_DIR/training.log"

TRAINING_EXIT_CODE=$?

echo ""
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo "📈 Training log: $OUTPUT_DIR/training.log"
    echo "🎯 Model outputs: models/trained/"
    echo "📊 Results: $OUTPUT_DIR/"
    
    # Show summary if available
    if [ -f "$OUTPUT_DIR/training_summary.json" ]; then
        echo ""
        echo "📋 Training Summary:"
        cat "$OUTPUT_DIR/training_summary.json" | python -m json.tool
    fi
else
    echo "❌ Training failed with exit code: $TRAINING_EXIT_CODE"
    echo "📋 Check logs for details: $OUTPUT_DIR/training.log"
    exit $TRAINING_EXIT_CODE
fi

echo ""
echo "🏁 FL Training Pipeline Complete!"
