#!/usr/bin/env python3
"""
Trace generation script for LLaMA 3.1 8B model with straggler analysis.
Configuration is loaded from hardcoded workload card YAML file.

Usage:
  Single GPU:
    python llama_trace_gen.py
"""

import os
import argparse
import yaml
import functools
import gzip
import json
import torch
import torch.distributed as dist
from torch.profiler import ExecutionTraceObserver, profile
from transformers import AutoModelForCausalLM, AutoTokenizer


def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        # Single GPU fallback
        rank = 0
        world_size = 1
        local_rank = 0
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return rank, world_size, local_rank


def load_llama_model(model_name_or_path, device):
    """Load LLaMA model and tokenizer."""
    print(f"Loading model from: {model_name_or_path}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=None,
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def load_workload_config(yaml_path):
    """Load configuration from workload card YAML file."""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check environment variable first
    env_model_path = os.environ.get('model_path')
    if env_model_path:
        model_path = env_model_path
    else:
        hf_url = config.get('hf_url', '')
        if hf_url:
            # Convert URL format to HuggingFace model ID (e.g., meta-llama/Llama-3.1-8B-Instruct)
            model_path = '/'.join(hf_url.split('/')[-2:])
        else:
            model_path = None
    
    # Store entire config plus extracted model_path
    config['model_path'] = model_path
    return config


def trace_handler(prof):
    """Callback to save kineto traces per rank."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    trace_file = f"kineto_trace_{rank}.json"
    prof.export_chrome_trace(trace_file)
    print(f"Rank {rank}: Saved kineto trace to {trace_file}")


def get_c4_data_iterator(file_path, tokenizer, batch_size, seq_length, rank, world_size, device):
    """Iterate over C4 dataset and yield batches of tokens."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"C4 dataset not found at: {file_path}")
        
    print(f"Rank {rank}: Loading C4 data from {file_path}")
    
    while True:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            buffer_tokens = []
            for i, line in enumerate(f):
                # Simple sharding based on line number
                if i % world_size != rank:
                    continue
                
                try:
                    data = json.loads(line)
                    text = data.get('text', '')
                    if not text:
                        continue
                        
                    # Encode text
                    tokens = tokenizer.encode(text, add_special_tokens=True)
                    buffer_tokens.extend(tokens)
                    
                    # Yield batches when we have enough tokens
                    while len(buffer_tokens) >= batch_size * seq_length:
                        batch_flat = buffer_tokens[:batch_size * seq_length]
                        buffer_tokens = buffer_tokens[batch_size * seq_length:]
                        
                        yield torch.tensor(batch_flat, dtype=torch.long, device=device).view(batch_size, seq_length)
                        
                except Exception as e:
                    print(f"Rank {rank}: Error processing line: {e}")
                    continue


def main():
    parser = argparse.ArgumentParser(description="Trace generation script for LLaMA 3.1 8B model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="../trace_collection/llama-3.1-8b-1GPU-DP-perlmutter/llama-3.1-8b-1GPU-DP-perlmutter.yaml",
        help="Path to the workload card YAML file"
    )
    args = parser.parse_args()
    
    workload_card_path = args.config
    
    # Setup distributed first to get rank
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    # Load config from workload card
    workload_config = load_workload_config(workload_card_path)
    if rank == 0:
        print(f"Loaded configuration from workload card: {workload_card_path}")
    
    # Get all settings from workload card
    batch_size = workload_config['workload']['data']['batch_size']
    seq_length = workload_config['workload']['data']['seq_len']
    num_iterations = workload_config['workload']['model']['iteration']
    model_path = workload_config['model_path']
    
    if not model_path:
        raise ValueError("Model path not found in workload card hf_url")
    
    # Output directory is same as workload card directory
    output_dir = os.path.dirname(os.path.abspath(workload_card_path))
    
    if rank == 0:
        print(f"Configuration:")
        print(f"  Model path: {model_path}")
        print(f"  World size: {world_size}")
        print(f"  Batch size per GPU: {batch_size}")
        print(f"  Sequence length: {seq_length}")
        print(f"  Num iterations: {num_iterations}")
        print(f"  Device: {device}")
        print(f"  Output dir: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)
    
    # Load LLaMA model
    model, tokenizer = load_llama_model(model_path, device)
    
    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    
    # Wrap with FSDP for multi-GPU to save memory
    if world_size > 1:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
        
        # Simple auto-wrap policy
        my_auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=1000000
        )
        
        model = FSDP(
            model,
            auto_wrap_policy=my_auto_wrap_policy,
            device_id=torch.cuda.current_device()
        )
    
    # Setup ExecutionTraceObserver
    et_file = f"torch_et_{rank}.json"
    et = ExecutionTraceObserver()
    et.register_callback(et_file)
    
    # Setup C4 data loader
    c4_file_path = "/pscratch/sd/k/kg597/c4_dataset/en/c4-train.00000-of-01024.json.gz"
    data_iterator = get_c4_data_iterator(
        c4_file_path, tokenizer, batch_size, seq_length, rank, world_size, device
    )
    
    if rank == 0:
        print(f"\nStarting training with profiling...")
    
    # Setup profiler with schedule
    with profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=0,     # no wait iterations
            warmup=5,   # 5 warmup iterations
            active=1    # profile 1 iteration (iteration 5)
        ),
        record_shapes=True,
        on_trace_ready=trace_handler
    ) as prof:
        
        for iteration in range(num_iterations):
            # Start ET at iteration 5 (when profiling starts)
            if iteration == 5:
                et.start()
                if rank == 0:
                    print(f"Started execution trace at iteration {iteration}")
            
            # Get batch from C4 dataset
            input_ids = next(data_iterator)
            
            # Forward pass (using labels triggers loss computation)
            outputs = model(
                input_ids=input_ids,
                labels=input_ids
            )
            
            # Backward pass (triggers gradient all-reduce in DDP)
            outputs.loss.backward()
            
            # Clear gradients to prevent accumulation across iterations
            model.zero_grad()
            
            # Synchronization barrier to measure straggler effect
            if world_size > 1:
                dist.barrier()
            
            # Stop ET at iteration 6 (after profiling)
            if iteration == 6:
                et.stop()
                if rank == 0:
                    print(f"Stopped execution trace at iteration {iteration}")
            
            # Must call prof.step() at end of each iteration
            prof.step()
            
            if rank == 0:
                print(f"Iteration {iteration}: loss={outputs.loss.item():.4f}")
    
    # Cleanup
    et.unregister_callback()
    
    if world_size > 1:
        dist.destroy_process_group()
    
    print(f"Rank {rank}: Traces saved:")
    print(f"  - {et_file}")
    print(f"  - kineto_trace_{rank}.json")
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Trace generation complete!")
        print(f"{'='*60}")
        print(f"\nTo analyze communication calls:")
        print(f"  python ../../tools/main.py --trace . --metric coll_call_num")
        print(f"\nTo analyze straggler delay:")
        print(f"  python ../../tools/main.py --trace . --metric straggler_delay")


if __name__ == "__main__":
    main()
