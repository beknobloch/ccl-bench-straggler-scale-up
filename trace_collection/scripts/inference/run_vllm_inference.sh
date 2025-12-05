#!/bin/bash
# Run vLLM Inference with NSYS Profiling
# Usage: ./run_vllm_inference.sh <model_name> <tp_size> <experiment_name>

set -e

MODEL_NAME=${1:-"meta-llama/Llama-3.1-8B"}
TP_SIZE=${2:-4}
EXPERIMENT_NAME=${3:-"llama-3.1-8b-vllm-tp4-4gpu"}
BATCH_SIZE=${4:-2}
INPUT_LEN=${5:-512}
OUTPUT_LEN=${6:-128}
NUM_PROMPTS=${7:-10}

# Base directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR="$(dirname $(dirname $SCRIPT_DIR))"
TRACE_DIR="${BASE_DIR}/${EXPERIMENT_NAME}"

echo "=========================================="
echo "vLLM Inference Experiment"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Tensor Parallel Size: $TP_SIZE"
echo "Experiment: $EXPERIMENT_NAME"
echo "Trace Directory: $TRACE_DIR"
echo "=========================================="

# Create trace directory
mkdir -p "$TRACE_DIR"

# Distributed environment variables to prevent NCCL hangs
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export NCCL_SOCKET_IFNAME=hsn  # Use high-speed network interface (Perlmutter specific)
export NCCL_DEBUG=INFO         # Enable NCCL logging for debugging
export VLLM_NCCL_SO_PATH=""    # Use system NCCL if needed

echo "Distributed Config:"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  NCCL_SOCKET_IFNAME: $NCCL_SOCKET_IFNAME"

# NSYS profiling configuration
NSYS_OUTPUT="${TRACE_DIR}/nsys"
# Added --trace-fork-before-exec=true to capture vLLM worker processes
NSYS_OPTS="--trace=cuda,nvtx,osrt,cudnn,cublas --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-memory-usage=true --trace-fork-before-exec=true"

echo "Starting NSYS profiling with vLLM inference..."

# Run with NSYS profiling
# NOTE: vLLM manages its own distributed workers, so we DO NOT use torchrun
# We launch a single python process, and vLLM spawns the workers.
# NSYS with --trace-fork-before-exec=true will capture all of them.

echo "Launching vLLM with TP=$TP_SIZE..."

nsys profile $NSYS_OPTS -o "${NSYS_OUTPUT}" \
    python3 "$SCRIPT_DIR/vllm_inference_tp.py" \
        --model "$MODEL_NAME" \
        --tensor-parallel-size $TP_SIZE \
        --batch-size $BATCH_SIZE \
        --input-len $INPUT_LEN \
        --output-len $OUTPUT_LEN \
        --num-prompts $NUM_PROMPTS \
        --output-dir "$TRACE_DIR" \
        --warmup-iters 2 \
        --profile-iters 3

echo "=========================================="
echo "Profiling Complete!"
echo "Traces saved to: $TRACE_DIR"
echo "=========================================="
echo "PyTorch traces:"
ls -lh "$TRACE_DIR"/*.json 2>/dev/null || echo "  No JSON traces found"
echo ""
echo "NSYS traces:"
ls -lh "$TRACE_DIR"/*.nsys-rep 2>/dev/null || echo "  No NSYS traces found"
echo "=========================================="
