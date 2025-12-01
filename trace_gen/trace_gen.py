#!/usr/bin/env python3
"""
Straightforward trace generator using PARAM as a library plus torch profiler.

- Reads a workload card under trace_collection/.
- Supports training (forward+backward) and inference (forward only).
- Collects PyTorch Execution Trace (`torch_et_<rank>.json`) and Kineto trace
  (`kineto_trace_<rank>.json`) side by side with the workload card so tools/main.py
  can consume them.
- Multi-GPU friendly: launch with torchrun; each rank writes its own traces.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import yaml
from torch.autograd.profiler import record_function
from torch.profiler import ExecutionTraceObserver, profile

# PARAM imports (installed as param_bench)
from param_bench.train.compute.python.lib import pytorch as lib_pytorch
from param_bench.train.compute.python.lib.config import BenchmarkConfig
from param_bench.train.compute.python.lib.init_helper import load_modules
from param_bench.train.compute.python.lib.pytorch.benchmark import make_default_benchmark
from param_bench.train.compute.python.lib.pytorch.config_util import (
    ExecutionPass,
    get_benchmark_options,
    OpExecutionMode,
)
from param_bench.train.compute.python.workloads import pytorch as workloads_pytorch

PRECISION_MAP = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp16": "float16",
    "float16": "float16",
    "fp32": "float",
    "float32": "float",
}

MODEL_HIDDEN_SIZE = {
    "llama-3.1-8b": 4096,
    "llama-3.1-70b": 8192,
    "llama-3-8b": 4096,
    "deepseek_v2": 4096,
}


def load_card(path: Path) -> Dict[str, Any]:
    with path.open("r") as handle:
        return yaml.safe_load(handle)


def get_rank_and_device() -> Tuple[int, int, torch.device]:
    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    return rank, world_size, device


def normalize_precision(raw: str | None) -> str:
    return PRECISION_MAP.get((raw or "fp32").lower(), "float")


def infer_hidden_size(card: Dict[str, Any]) -> int:
    model_family = (
        card.get("workload", {}).get("model", {}).get("model_family", "").lower().replace(" ", "")
    )
    return MODEL_HIDDEN_SIZE.get(model_family, 4096)


def build_param_config(batch_size: int, seq_len: int, hidden: int, dtype: str, requires_grad: bool) -> Dict[str, Any]:
    """Tiny transformer-like workload: one Linear (weights) and one matmul."""
    seq_dim = max(1, seq_len)
    token_dim = max(1, batch_size)
    common_tensor = {"type": "tensor", "dtype": dtype, "requires_grad": requires_grad}
    return {
        "torch.nn.Linear": {
            "input_data_generator": "PyTorch:DefaultDataGenerator",
            "config": [
                {
                    "build": [
                        {
                            "args": [
                                {"type": "int", "value": hidden},
                                {"type": "int", "value": hidden},
                            ],
                            "kwargs": {"bias": {"type": "bool", "value": False}},
                        }
                    ],
                    "input": [
                        {
                            "args": [
                                {**common_tensor, "shape": [token_dim, seq_dim, hidden]},
                            ]
                        }
                    ],
                }
            ],
        },
        "torch.mm": {
            "input_data_generator": "PyTorch:DefaultDataGenerator",
            "config": [
                {
                    "input": [
                        {
                            "args": [
                                {**common_tensor, "shape": [token_dim * seq_dim, hidden]},
                                {**common_tensor, "shape": [hidden, hidden]},
                            ]
                        }
                    ]
                }
            ],
        },
    }


def build_run_options(device: torch.device, phase: str, warmup: int, iteration: int) -> Dict[str, Any]:
    opts = get_benchmark_options()
    opts["device"] = str(device)
    opts["warmup"] = warmup
    opts["iteration"] = iteration
    opts["op_exec_mode"] = OpExecutionMode("discrete")
    opts["pass_type"] = ExecutionPass.BACKWARD if phase == "training" else ExecutionPass.FORWARD
    opts["run_ncu"] = False
    opts["run_nsys"] = False
    return opts


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Kineto + ET traces using PARAM.")
    parser.add_argument("--workload-card", required=True, type=Path, help="Path to workload YAML under trace_collection/")
    parser.add_argument("--task", choices=["training", "inference"], help="Override workload.model.phase")
    parser.add_argument("--warmup", type=int, default=1, help="Profiler warmup iterations")
    parser.add_argument("--iteration", type=int, help="Measured iterations (fallback to card or 10)")
    args = parser.parse_args()

    card = load_card(args.workload_card)
    workload = card.get("workload", {})
    model_cfg = workload.get("model", {})
    data_cfg = workload.get("data", {})

    phase = (args.task or model_cfg.get("phase", "training")).lower()
    if phase not in {"training", "inference"}:
        raise ValueError("phase must be training or inference")

    batch_size = int(data_cfg.get("batch_size", 1))
    seq_len = int(data_cfg.get("seq_len", 1024))
    hidden_size = infer_hidden_size(card)
    precision = normalize_precision(model_cfg.get("precision"))
    iteration = args.iteration or int(model_cfg.get("iteration", 10))

    rank, world_size, device = get_rank_and_device()
    output_dir = args.workload_card.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build PARAM benchmark
    load_modules(lib_pytorch)
    load_modules(workloads_pytorch)
    run_options = build_run_options(device, phase, args.warmup, iteration)
    bench_cfg = BenchmarkConfig(run_options)
    bench_cfg.load(build_param_config(batch_size, seq_len, hidden_size, precision, phase == "training"))
    benchmark = make_default_benchmark(bench_cfg)

    # Trace file names
    prefix = f"{args.workload_card.stem}_{phase}_rank{rank}"
    et_file = output_dir / f"{prefix}_et.json"
    kineto_file = output_dir / f"{prefix}_trace.json"

    # Setup ET
    et = ExecutionTraceObserver()
    et.register_callback(str(et_file))
    et.start()

    # Profile + run benchmark
    use_cuda = device.type == "cuda"
    with profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA] if use_cuda else [torch.profiler.ProfilerActivity.CPU],
        schedule=torch.profiler.schedule(wait=0, warmup=args.warmup, active=1),
        record_shapes=True,
    ) as prof:
        with record_function(f"[param|{device}]"):
            benchmark.run()
        prof.export_chrome_trace(str(kineto_file))

    et.stop()
    et.unregister_callback()

    # Print summary for easy discovery
    print(f"[Rank {rank}/{world_size}] ET trace: {et_file}")
    print(f"[Rank {rank}/{world_size}] Kineto trace: {kineto_file}")
    if world_size > 1:
        torch.distributed.barrier()


if __name__ == "__main__":
    main()

# Example command (inference, single GPU):
#   python trace_gen/trace_gen.py --workload-card trace_collection/llama-3.1-8b-1GPU-DP-perlmutter/llama-3.1-8b-1GPU-DP-perlmutter-inference.yaml
# Multi-GPU with torchrun (each rank writes its own trace):
#   torchrun --nproc_per_node=4 trace_gen/trace_gen.py --workload-card trace_collection/llama-3.1-8b-torchtitan-perlmutter/llama-3.1-8b-torchtitan-perlmutter.yaml --task training


# hf auth login
# export HF_HOME=$PSCRATCH/huggingface
# huggingface-cli download meta-llama/CodeLlama-34b-hf