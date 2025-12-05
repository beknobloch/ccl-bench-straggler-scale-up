#!/bin/bash
# Analyze straggler metrics across all collected traces

set -e

TRACE_BASE_DIR=${1:-"../trace_collection"}
OUTPUT_CSV=${2:-"straggler_metrics.csv"}

echo "=========================================="
echo "Straggler Metric Analysis"
echo "=========================================="
echo "Trace Directory: $TRACE_BASE_DIR"
echo "Output CSV: $OUTPUT_CSV"
echo "=========================================="

# Create CSV header
echo "experiment,straggler_delay,straggler_slowdown" > "$OUTPUT_CSV"

# Find all trace directories (those containing kineto_trace files)
for trace_dir in "$TRACE_BASE_DIR"/*/; do
    # Check if directory contains kineto trace files
    if ls "$trace_dir"/kineto_trace_*.json 1> /dev/null 2>&1; then
        experiment_name=$(basename "$trace_dir")
        echo "Processing: $experiment_name"
        
        # Calculate straggler_delay
        delay=$(python ../tools/main.py --trace "$trace_dir" --metric straggler_delay 2>/dev/null || echo "N/A")
        
        # Calculate straggler_slowdown
        slowdown=$(python ../tools/main.py --trace "$trace_dir" --metric straggler_slowdown 2>/dev/null || echo "N/A")
        
        echo "  Straggler Delay: $delay"
        echo "  Straggler Slowdown: $slowdown"
        
        # Append to CSV
        echo "$experiment_name,$delay,$slowdown" >> "$OUTPUT_CSV"
    fi
done

echo "=========================================="
echo "Analysis Complete!"
echo "Results saved to: $OUTPUT_CSV"
echo "=========================================="
cat "$OUTPUT_CSV"
echo "=========================================="
