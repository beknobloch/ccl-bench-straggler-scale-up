#!/bin/bash
# Run TorchTitan Training with NSYS Profiling
# Usage: ./run_torchtitan.sh <model_name> <parallelization> <experiment_name>

set -e

MODEL_NAME=${1:-"meta-llama/Llama-3.1-8B"}
PARALLELIZATION=${2:-"fsdp"}  # fsdp, tp, or ddp
EXPERIMENT_NAME=${3:-"llama-3.1-8b-torchtitan-fsdp-4gpu"}
BATCH_SIZE=${4:-2}
SEQ_LEN=${5:-512}
NUM_GPUS=${6:-4}

# Directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR="$(dirname $(dirname $SCRIPT_DIR))"
TRACE_DIR="${BASE_DIR}/${EXPERIMENT_NAME}"

echo "=========================================="
echo "TorchTitan Training Experiment"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Parallelization: $PARALLELIZATION"
echo "Experiment: $EXPERIMENT_NAME"
echo "GPUs: $NUM_GPUS"
echo "Trace Directory: $TRACE_DIR"
echo "=========================================="

# Create trace directory
mkdir -p "$TRACE_DIR"

# Launch with torchrun
echo "Launching TorchTitan training with $NUM_GPUS GPUs..."

torchrun --nproc_per_node=$NUM_GPUS \
    "$SCRIPT_DIR/torchtitan_train.py" \
        --model "$MODEL_NAME" \
        --parallelization "$PARALLELIZATION" \
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
