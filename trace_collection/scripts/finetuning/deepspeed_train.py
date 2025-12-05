#!/usr/bin/env python3
"""
DeepSpeed Training Script with Profiling for Straggler Analysis
Supports ZeRO-2, ZeRO-3 on 4 GPUs
"""

import argparse
import os
import torch
import deepspeed
from torch.profiler import ExecutionTraceObserver, profile, ProfilerActivity, schedule


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


def get_model_and_tokenizer(model_name):
    """Load model and tokenizer"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer


def get_deepspeed_config(zero_stage, batch_size):
    """Generate DeepSpeed configuration"""
    config = {
        "train_batch_size": batch_size,
        "gradient_accumulation_steps": 1,
        "steps_per_print": 1,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-5,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        },
        "fp16": {
            "enabled": False
        },
        "bf16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": zero_stage,
        }
    }
    
    if zero_stage == 3:
        config["zero_optimization"]["stage3_param_persistence_threshold"] = 1e4
        config["zero_optimization"]["stage3_max_live_parameters"] = 1e9
        config["zero_optimization"]["stage3_prefetch_bucket_size"] = 5e7
    
    return config


def main():
    parser = argparse.ArgumentParser(description="DeepSpeed Training with Profiling")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name")
    parser.add_argument("--zero-stage", type=int, default=2, choices=[0, 1, 2, 3], 
                       help="DeepSpeed ZeRO stage")
    parser.add_argument("--batch-size", type=int, default=2, help="Global batch size")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--num-iters", type=int, default=5, help="Total iterations")
    parser.add_argument("--warmup-iters", type=int, default=2, help="Warmup iterations")
    parser.add_argument("--profile-iters", type=int, default=3, help="Profile iterations")
    parser.add_argument("--output-dir", type=str, default="./traces", help="Output directory")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank (set by DeepSpeed)")
    args = parser.parse_args()
    
    # DeepSpeed initialization
    deepspeed.init_distributed()
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    
    print(f"[Rank {rank}] Starting DeepSpeed training with ZeRO-{args.zero_stage}")
    
    # Setup profiling
    et, trace_handler = setup_profiling(rank, args.output_dir)
    
    # Load model and tokenizer
    model, tokenizer = get_model_and_tokenizer(args.model)
    
    # DeepSpeed config
    ds_config = get_deepspeed_config(args.zero_stage, args.batch_size)
    
    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
    )
    
    # Create dummy dataset
    print(f"[Rank {rank}] Creating dummy dataset...")
    dummy_text = "This is a sample training text. " * 50
    inputs = tokenizer(
        [dummy_text] * (args.batch_size // model_engine.world_size),
        return_tensors="pt",
        max_length=args.seq_len,
        padding="max_length",
        truncation=True,
    )
    input_ids = inputs["input_ids"].to(model_engine.device)
    attention_mask = inputs["attention_mask"].to(model_engine.device)
    
    # Warmup
    print(f"[Rank {rank}] Warmup iterations...")
    model_engine.train()
    for i in range(args.warmup_iters):
        outputs = model_engine(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        model_engine.backward(loss)
        model_engine.step()
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
            outputs = model_engine(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            model_engine.backward(loss)
            model_engine.step()
            prof.step()
            if rank == 0:
                print(f"[Rank {rank}] Profile iter {i+1}/{args.profile_iters}, loss: {loss.item():.4f}")
    
    et.stop()
    et.unregister_callback()
    
    print(f"[Rank {rank}] Training profiling completed!")
    print(f"[Rank {rank}] Traces saved to {args.output_dir}")


if __name__ == "__main__":
    main()
