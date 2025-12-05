#!/bin/bash
# Run DeepSpeed Training with NSYS Profiling
# Usage: ./run_deepspeed.sh <model_name> <zero_stage> <experiment_name>

set -e

MODEL_NAME=${1:-"meta-llama/Llama-3.1-8B"}
ZERO_STAGE=${2:-2}
EXPERIMENT_NAME=${3:-"llama-3.1-8b-deepspeed-zero2-4gpu"}
BATCH_SIZE=${4:-8}  # Global batch size
SEQ_LEN=${5:-512}
NUM_GPUS=${6:-4}

# Directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR="$(dirname $(dirname $SCRIPT_DIR))"
TRACE_DIR="${BASE_DIR}/${EXPERIMENT_NAME}"

echo "=========================================="
echo "DeepSpeed Training Experiment"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "ZeRO Stage: $ZERO_STAGE"
echo "Experiment: $EXPERIMENT_NAME"
echo "GPUs: $NUM_GPUS"
echo "Trace Directory: $TRACE_DIR"
echo "=========================================="

# Create trace directory
mkdir -p "$TRACE_DIR"

# Launch with DeepSpeed
echo "Launching DeepSpeed training with ZeRO-$ZERO_STAGE on $NUM_GPUS GPUs..."

deepspeed --num_gpus=$NUM_GPUS \
    "$SCRIPT_DIR/deepspeed_train.py" \
        --model "$MODEL_NAME" \
        --zero-stage $ZERO_STAGE \
        --batch-size $BATCH_SIZE \
        --seq-len $SEQ_LEN \
        --num-iters 5 \
        --warmup-iters 2 \
        --profile-iters 3 \
        --output-dir "$TRACE_DIR"

echo "=========================================="
echo "Training Complete!"
echo "Traces saved to: $TRACE_DIR"
echo "=========================================="
ls -lh "$TRACE_DIR"/*.json 2>/dev/null || echo "No traces found"
echo "=========================================="
