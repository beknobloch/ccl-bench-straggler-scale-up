#!/bin/bash
# Master script to collect all traces for an experiment
# Combines NSYS, PyTorch ET, and Kineto traces

set -e

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <script_type> <script_path> <experiment_name> [additional args...]"
    echo "  script_type: inference, torchtitan, or deepspeed"
    echo "  script_path: path to the Python script to profile"
    echo "  experiment_name: name for this experiment (used for trace directory)"
    echo ""
    echo "Examples:"
    echo "  $0 inference vllm_inference_tp.py llama-8b-tp4 --model meta-llama/Llama-3.1-8B --tensor-parallel-size 4"
    echo "  $0 torchtitan torchtitan_train.py llama-8b-fsdp --model meta-llama/Llama-3.1-8B --parallelization fsdp"
    echo "  $0 deepspeed deepspeed_train.py llama-8b-zero2 --model meta-llama/Llama-3.1-8B --zero-stage 2"
    exit 1
fi

SCRIPT_TYPE=$1
SCRIPT_PATH=$2
EXPERIMENT_NAME=$3
shift 3
EXTRA_ARGS="$@"

# Directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR="$(dirname $(dirname $SCRIPT_DIR))"
TRACE_DIR="${BASE_DIR}/${EXPERIMENT_NAME}"

echo "=========================================="
echo "Comprehensive Trace Collection"
echo "=========================================="
echo "Script Type: $SCRIPT_TYPE"
echo "Script: $SCRIPT_PATH"
echo "Experiment: $EXPERIMENT_NAME"
echo "Trace Directory: $TRACE_DIR"
echo "Extra Args: $EXTRA_ARGS"
echo "=========================================="

# Create trace directory
mkdir -p "$TRACE_DIR"

# Determine the launcher command based on script type
case $SCRIPT_TYPE in
    inference)
        # For vLLM inference, determine if we need torchrun
        if [[ "$EXTRA_ARGS" == *"--tensor-parallel-size 1"* ]] || [[ "$EXTRA_ARGS" != *"--tensor-parallel-size"* ]]; then
            # Single GPU
            echo "Running single-GPU inference..."
            python3 "$SCRIPT_PATH" --output-dir "$TRACE_DIR" $EXTRA_ARGS
        else
            # Multi-GPU with torchrun
            # Extract TP size
            TP_SIZE=$(echo "$EXTRA_ARGS" | grep -oP '(?<=--tensor-parallel-size )\d+' || echo "4")
            echo "Running multi-GPU inference with TP=$TP_SIZE..."
            torchrun --nproc_per_node=$TP_SIZE \
                "$SCRIPT_PATH" --output-dir "$TRACE_DIR" $EXTRA_ARGS
        fi
        ;;
    
    torchtitan)
        # TorchTitan with torchrun
        NUM_GPUS=$(echo "$EXTRA_ARGS" | grep -oP '(?<=--num-gpus )\d+' || echo "4")
        echo "Running TorchTitan training with $NUM_GPUS GPUs..."
        torchrun --nproc_per_node=$NUM_GPUS \
            "$SCRIPT_PATH" --output-dir "$TRACE_DIR" $EXTRA_ARGS
        ;;
    
    deepspeed)
        # DeepSpeed with deepspeed launcher
        NUM_GPUS=$(echo "$EXTRA_ARGS" | grep -oP '(?<=--num-gpus )\d+' || echo "4")
        echo "Running DeepSpeed training with $NUM_GPUS GPUs..."
        deepspeed --num_gpus=$NUM_GPUS \
            "$SCRIPT_PATH" --output-dir "$TRACE_DIR" $EXTRA_ARGS
        ;;
    
    *)
        echo "ERROR: Unknown script type: $SCRIPT_TYPE"
        echo "Must be one of: inference, torchtitan, deepspeed"
        exit 1
        ;;
esac

echo "=========================================="
echo "Trace Collection Complete!"
echo "=========================================="
echo "Trace directory: $TRACE_DIR"
echo ""
echo "Collected traces:"
ls -lh "$TRACE_DIR" 2>/dev/null || echo "No files found"
echo ""
echo "Next steps:"
echo "  1. Verify traces are present (kineto_trace_*.json, torch_et_*.json)"
echo "  2. (Optional) Run NSYS profiling separately if needed"
echo "  3. Analyze with: python tools/main.py --trace $TRACE_DIR --metric straggler_delay"
echo "=========================================="
