#!/bin/bash
# Generic NSYS Profiling Wrapper for Distributed Training/Inference
# This script wraps any distributed command with NSYS profiling per rank

set -e

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <num_gpus> <output_dir> <command...>"
    echo "Example: $0 4 ./traces torchrun --nproc_per_node=4 train.py"
    exit 1
fi

NUM_GPUS=$1
OUTPUT_DIR=$2
shift 2
COMMAND="$@"

echo "=========================================="
echo "NSYS Distributed Profiling"
echo "=========================================="
echo "GPUs: $NUM_GPUS"
echo "Output Directory: $OUTPUT_DIR"
echo "Command: $COMMAND"
echo "=========================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# NSYS profiling options
# - Capture CUDA kernels, NVTX markers, OS runtime, cuDNN, cuBLAS
# - Capture NCCL operations  
# - Track CUDA memory usage
NSYS_OPTS="--trace=cuda,nvtx,osrt,cudnn,cublas,nccl --cuda-memory-usage=true --capture-range=cudaProfilerApi"

# For multi-GPU profiling, we need to profile each rank separately
# This is typically done by checking RANK or LOCAL_RANK environment variable
# and launching nsys only for that specific rank

# However, a simpler approach for torchrun/deepspeed is to use nsys with --trace-fork-before-exec
# which will trace all child processes

echo "Launching command with NSYS profiling..."
echo "Trace files will be saved to: $OUTPUT_DIR/nsys_*.nsys-rep"

# Run with NSYS
nsys profile $NSYS_OPTS \
    --trace-fork-before-exec=true \
    --output="$OUTPUT_DIR/nsys" \
    $COMMAND

echo "=========================================="
echo "NSYS Profiling Complete!"
echo "=========================================="
echo "Trace files:"
ls -lh "$OUTPUT_DIR"/*.nsys-rep 2>/dev/null || echo "No NSYS traces found"
echo ""
echo "To view traces, use:"
echo "  nsys-ui (GUI)"
echo "  nsys stats <trace_file>"
echo "  nsys export --type sqlite <trace_file>"
echo "=========================================="
