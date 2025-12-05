#!/usr/bin/env python3
"""
TorchTitan Training Script with Profiling for Straggler Analysis
Supports FSDP, TP, and various parallelization strategies on 4 GPUs
"""

import argparse
import os
import torch
import torch.distributed as dist
from torch.profiler import ExecutionTraceObserver, profile, ProfilerActivity, schedule


def setup_distributed():
    """Initialize distributed training"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def setup_profiling(rank, output_dir):
    """Setup PyTorch profiling"""
    os.makedirs(output_dir, exist_ok=True)
    
    et = ExecutionTraceObserver()
    et_path = os.path.join(output_dir, f"torch_et_{rank}.json")
    et.register_callback(et_path)
    
    def trace_handler(prof):
        kineto_path = os.path.join(output_dir, f"kineto_trace_{rank}.json")
        prof.export_chrome_trace(kineto_path)
        print(f"[Rank {rank}] Saved Kineto trace to {kineto_path}")
    
    return et, trace_handler


def get_model_and_tokenizer(model_name, dtype=torch.bfloat16):
    """Load model and tokenizer"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("ERROR: transformers not installed. Install with: pip install transformers")
        return None, None
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=None,  # We'll handle device placement
    )
    
    return model, tokenizer


def apply_fsdp(model, rank):
    """Apply FSDP to the model"""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy
    
    print(f"[Rank {rank}] Applying FSDP...")
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="TorchTitan Training with Profiling")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name or path")
    parser.add_argument("--parallelization", type=str, default="fsdp", 
                       choices=["fsdp", "tp", "ddp"], help="Parallelization strategy")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size (if using TP)")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per GPU")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--num-iters", type=int, default=5, help="Total training iterations")
    parser.add_argument("--warmup-iters", type=int, default=2, help="Warmup iterations")
    parser.add_argument("--profile-iters", type=int, default=3, help="Iterations to profile")
    parser.add_argument("--output-dir", type=str, default="./traces", help="Output directory")
    parser.add_argument("--dtype", type=str, default="bfloat16", 
                       choices=["float16", "bfloat16", "float32"], help="Model dtype")
    args = parser.parse_args()
    
    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    
    print(f"[Rank {rank}/{world_size}] Starting TorchTitan training...")
    print(f"[Rank {rank}] Parallelization: {args.parallelization}")
    
    # Setup profiling
    et, trace_handler = setup_profiling(rank, args.output_dir)
    
    # Determine dtype
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]
    
    # Load model
    model, tokenizer = get_model_and_tokenizer(args.model, dtype)
    if model is None:
        return
    
    # Move to GPU
    device = torch.device(f"cuda:{local_rank}")
    model = model.to(device)
    
    # Apply parallelization
    if args.parallelization == "fsdp" and world_size > 1:
        model = apply_fsdp(model, rank)
    elif args.parallelization == "tp":
        print(f"[Rank {rank}] TP not yet implemented - using FSDP instead")
        if world_size > 1:
            model = apply_fsdp(model, rank)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    # Create dummy dataset
    print(f"[Rank {rank}] Creating dummy dataset...")
    dummy_text = "This is a sample training text. " * 50
    inputs = tokenizer(
        [dummy_text] * args.batch_size,
        return_tensors="pt",
        max_length=args.seq_len,
        padding="max_length",
        truncation=True,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Warmup
    print(f"[Rank {rank}] Warmup iterations...")
    model.train()
    for i in range(args.warmup_iters):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if rank == 0:
            print(f"[Rank {rank}] Warmup iter {i+1}/{args.warmup_iters}, loss: {loss.item():.4f}")
    
    # Profiling
    print(f"[Rank {rank}] Starting profiling...")
    et.start()
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=0, warmup=0, active=args.profile_iters, repeat=1),
        record_shapes=True,
        with_stack=True,
        on_trace_ready=trace_handler,
    ) as prof:
        for i in range(args.profile_iters):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            prof.step()
            if rank == 0:
                print(f"[Rank {rank}] Profile iter {i+1}/{args.profile_iters}, loss: {loss.item():.4f}")
    
    et.stop()
    et.unregister_callback()
    
    print(f"[Rank {rank}] Training profiling completed!")
    print(f"[Rank {rank}] Traces saved to {args.output_dir}")
    
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
