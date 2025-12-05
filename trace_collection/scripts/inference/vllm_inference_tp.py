#!/usr/bin/env python3
"""
vLLM Inference Script with Profiling for Straggler Analysis
Supports Tensor Parallelism (TP) on 4 GPUs
"""

import argparse
import json
import os
import torch
from torch.profiler import ExecutionTraceObserver, profile, ProfilerActivity, schedule


def setup_profiling(rank, output_dir):
    """Setup PyTorch profiling with Kineto and ExecutionTraceObserver"""
    os.makedirs(output_dir, exist_ok=True)
    
    # ExecutionTraceObserver for PyTorch ET traces
    et = ExecutionTraceObserver()
    et_path = os.path.join(output_dir, f"torch_et_{rank}.json")
    et.register_callback(et_path)
    
    # Kineto trace handler
    def trace_handler(prof):
        kineto_path = os.path.join(output_dir, f"kineto_trace_{rank}.json")
        prof.export_chrome_trace(kineto_path)
        print(f"[Rank {rank}] Saved Kineto trace to {kineto_path}")
    
    return et, trace_handler


def main():
    parser = argparse.ArgumentParser(description="vLLM Inference with Profiling")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name or path")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallelism degree")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--input-len", type=int, default=512, help="Input sequence length")
    parser.add_argument("--output-len", type=int, default=128, help="Output sequence length")
    parser.add_argument("--num-prompts", type=int, default=10, help="Number of prompts to process")
    parser.add_argument("--output-dir", type=str, default="./traces", help="Output directory for traces")
    parser.add_argument("--warmup-iters", type=int, default=2, help="Warmup iterations before profiling")
    parser.add_argument("--profile-iters", type=int, default=3, help="Iterations to profile")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"], help="Model dtype")
    args = parser.parse_args()
    
    # Import vLLM after argument parsing
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("ERROR: vLLM not installed. Install with: pip install vllm")
        return
    
    # Determine local rank for multi-GPU setups
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Setup profiling
    et, trace_handler = setup_profiling(local_rank, args.output_dir)
    
    print(f"[Rank {local_rank}] Initializing vLLM with model: {args.model}")
    print(f"[Rank {local_rank}] Tensor Parallel Size: {args.tensor_parallel_size}")
    
    # Initialize vLLM engine
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        max_model_len=args.input_len + args.output_len,
        enforce_eager=True,  # Disable CUDA graphs for better profiling
    )
    
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.output_len,
    )
    
    # Generate dummy prompts
    prompts = [f"Summarize the following text in detail: {'sample ' * args.input_len}" for _ in range(args.num_prompts)]
    
    print(f"[Rank {local_rank}] Starting warmup ({args.warmup_iters} iterations)...")
    # Warmup
    for i in range(args.warmup_iters):
        llm.generate(prompts[:args.batch_size], sampling_params)
        print(f"[Rank {local_rank}] Warmup iteration {i+1}/{args.warmup_iters} completed")
    
    print(f"[Rank {local_rank}] Starting profiling ({args.profile_iters} iterations)...")
    
    # Start profiling
    et.start()
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=0, warmup=0, active=args.profile_iters, repeat=1),
        record_shapes=True,
        with_stack=True,
        on_trace_ready=trace_handler,
    ) as prof:
        for i in range(args.profile_iters):
            print(f"[Rank {local_rank}] Profiling iteration {i+1}/{args.profile_iters}...")
            outputs = llm.generate(prompts[:args.batch_size], sampling_params)
            prof.step()
            print(f"[Rank {local_rank}] Generated {len(outputs)} outputs")
    
    et.stop()
    et.unregister_callback()
    
    print(f"[Rank {local_rank}] Profiling completed! Traces saved to {args.output_dir}")
    print(f"[Rank {local_rank}] - PyTorch ET: torch_et_{local_rank}.json")
    print(f"[Rank {local_rank}] - Kineto: kineto_trace_{local_rank}.json")


if __name__ == "__main__":
    main()
