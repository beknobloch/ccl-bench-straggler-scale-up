#!/usr/bin/env python3
"""
Trace generator using TorchTitan + torch.profiler.

- Reads a workload card under trace_collection/.
- Supports training (forward+backward) and inference (forward only).
- Collects PyTorch Execution Trace (`*_rank<N>_et.json`) and Kineto trace
  (`*_rank<N>_trace.json`) next to the workload card.
- Designed to run under torchrun for multi-GPU; TorchTitan is optional but
  parallel settings from the card are parsed and honored when possible.
"""

from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.distributed as dist
import yaml
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import ExecutionTraceObserver, ProfilerActivity, profile, schedule

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
except Exception:  # pragma: no cover - FSDP might be missing on CPU-only envs
    FSDP = None

PRECISION_TO_DTYPE = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}

MODEL_HIDDEN_SIZE = {
    "llama-3.1-8b": 4096,
    "llama-3.1-70b": 8192,
    "llama-3-8b": 4096,
    "deepseek_v2": 4096,
}


class TinyTransformerLM(nn.Module):
    """Lightweight transformer-ish block to exercise kernels without heavy deps."""

    def __init__(self, hidden_size: int, vocab_size: int = 32000) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        x = self.ffn(x)
        return self.lm_head(x)


def load_card(path: Path) -> Dict[str, Any]:
    with path.open("r") as handle:
        return yaml.safe_load(handle)


def normalize_precision(raw: str | None) -> torch.dtype:
    if raw is None:
        return torch.float32
    return PRECISION_TO_DTYPE.get(raw.lower(), torch.float32)


def infer_hidden_size(card: Dict[str, Any]) -> int:
    model_family = (
        card.get("workload", {})
        .get("model", {})
        .get("model_family", "")
        .lower()
        .replace(" ", "")
    )
    return MODEL_HIDDEN_SIZE.get(model_family, 4096)


def setup_distributed() -> Tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return rank, local_rank, world_size, device


def parse_parallel_cfg(card: Dict[str, Any]) -> Dict[str, int]:
    defaults = {"dp_replicate": 1, "dp_shard": 1, "tp": 1, "pp": 1, "cp": 1}
    user_cfg = card.get("Model-executor", {}).get("model_plan_parallelization", {}) or {}
    for key in defaults:
        defaults[key] = int(user_cfg.get(key, defaults[key]) or 1)
    return defaults


def maybe_init_torchtitan(parallel_cfg: Dict[str, int]) -> None:
    """Attempt to initialize TorchTitan parallel groups if available."""
    try:
        import torchtitan  # type: ignore
    except ImportError:
        print("[torchtitan] Not installed; continuing with torch.distributed only.")
        return

    params = {
        "tensor_parallel_size": parallel_cfg["tp"],
        "pipeline_parallel_size": parallel_cfg["pp"],
        "context_parallel_size": parallel_cfg["cp"],
        "data_parallel_size": parallel_cfg["dp_replicate"] * parallel_cfg["dp_shard"],
        "tp_size": parallel_cfg["tp"],
        "pp_size": parallel_cfg["pp"],
        "cp_size": parallel_cfg["cp"],
        "dp_size": parallel_cfg["dp_replicate"] * parallel_cfg["dp_shard"],
    }

    candidates = []
    parallel_mod = getattr(torchtitan, "parallel", None)
    for name in ("initialize_model_parallel", "init_model_parallel", "initialize_parallel_groups"):
        if parallel_mod is not None and hasattr(parallel_mod, name):
            candidates.append((f"torchtitan.parallel.{name}", getattr(parallel_mod, name)))
        if hasattr(torchtitan, name):
            candidates.append((f"torchtitan.{name}", getattr(torchtitan, name)))

    for label, fn in candidates:
        if fn is None:
            continue
        try:
            sig = inspect.signature(fn)
            kwargs = {arg: params[arg] for arg in sig.parameters if arg in params}
            fn(**kwargs)
            print(f"[torchtitan] Initialized via {label} with {kwargs}")
            return
        except Exception as exc:  # pragma: no cover - runtime environment specific
            print(f"[torchtitan] {label} failed ({exc}); will keep going.")

    print("[torchtitan] Present but no init function matched; skipping Titan init.")


def wrap_model_for_parallelism(
    model: nn.Module,
    device: torch.device,
    world_size: int,
    parallel_cfg: Dict[str, int],
) -> nn.Module:
    model = model.to(device)
    if world_size <= 1:
        return model

    if parallel_cfg["dp_shard"] > 1 and FSDP is not None:
        return FSDP(model)
    if parallel_cfg["dp_shard"] > 1 and FSDP is None:
        print("[warning] dp_shard>1 requested but torch.distributed.fsdp is unavailable; falling back to DDP.")

    device_id = device.index if device.type == "cuda" else None
    return DDP(model, device_ids=[device_id] if device_id is not None else None)


def generate_tokens(batch_size: int, seq_len: int, vocab_size: int, device: torch.device) -> torch.Tensor:
    return torch.randint(low=0, high=vocab_size - 1, size=(batch_size, seq_len), device=device)


def run_training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    dtype: torch.dtype,
    device_type: str,
) -> float:
    model.train()
    with torch.autocast(device_type=device_type, dtype=dtype, enabled=dtype != torch.float32):
        logits = model(input_ids)
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), input_ids.view(-1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(loss.detach())


def run_inference_step(
    model: nn.Module,
    input_ids: torch.Tensor,
    dtype: torch.dtype,
    device_type: str,
) -> float:
    model.eval()
    with torch.no_grad(), torch.autocast(device_type=device_type, dtype=dtype, enabled=dtype != torch.float32):
        logits = model(input_ids)
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), input_ids.view(-1))
    return float(loss.detach())


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Kineto + ET traces using TorchTitan + torch.profiler.")
    parser.add_argument("--workload-card", required=True, type=Path, help="Path to workload YAML under trace_collection/")
    parser.add_argument("--task", choices=["training", "inference"], help="Override workload.model.phase")
    parser.add_argument("--warmup", type=int, default=1, help="Profiler warmup iterations")
    parser.add_argument("--iteration", type=int, help="Measured iterations (defaults to workload.model.iteration or 10)")
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
    iterations = args.iteration or int(model_cfg.get("iteration", 10))
    iterations = max(iterations, args.warmup + 1)
    vocab_size = 32000

    rank, _, world_size, device = setup_distributed()
    parallel_cfg = parse_parallel_cfg(card)
    expected_world = (
        parallel_cfg["dp_replicate"]
        * parallel_cfg["dp_shard"]
        * parallel_cfg["tp"]
        * parallel_cfg["pp"]
        * parallel_cfg["cp"]
    )
    if expected_world and expected_world != 1 and expected_world != world_size:
        print(
            f"[warning] Product of parallel dims ({expected_world}) != world_size ({world_size}); "
            "continuing but check your launch command."
        )
    if any(parallel_cfg.get(dim, 1) > 1 for dim in ("tp", "pp", "cp")):
        print("[note] tp/pp/cp parsed from the card; this script keeps a single-module toy model.")

    maybe_init_torchtitan(parallel_cfg)

    output_dir = args.workload_card.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    model = TinyTransformerLM(hidden_size=hidden_size, vocab_size=vocab_size)
    model = wrap_model_for_parallelism(model, device, world_size, parallel_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    prefix = f"{args.workload_card.stem}_{phase}_rank{rank}"
    et_file = output_dir / f"{prefix}_et.json"
    kineto_file = output_dir / f"{prefix}_trace.json"

    et = ExecutionTraceObserver()
    et.register_callback(str(et_file))
    et.start()

    losses: list[float] = []
    try:
        use_cuda = device.type == "cuda"
        prof_activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if use_cuda else [])
        prof_schedule = schedule(wait=0, warmup=args.warmup, active=1)

        device_type = "cuda" if use_cuda else "cpu"
        with profile(activities=prof_activities, schedule=prof_schedule, record_shapes=True) as prof:
            for step in range(iterations):
                input_ids = generate_tokens(batch_size, seq_len, vocab_size, device)
                if phase == "training":
                    loss_val = run_training_step(model, optimizer, input_ids, precision, device_type)
                else:
                    loss_val = run_inference_step(model, input_ids, precision, device_type)
                losses.append(loss_val)
                if world_size > 1 and dist.is_initialized():
                    dist.barrier()
                prof.step()
            prof.export_chrome_trace(str(kineto_file))
    finally:
        et.stop()
        et.unregister_callback()

    if world_size > 1 and dist.is_initialized():
        dist.barrier()

    mean_loss = sum(losses) / max(1, len(losses))
    print(f"[Rank {rank}/{world_size}] phase={phase} mean_loss={mean_loss:.4f}")
    print(f"[Rank {rank}/{world_size}] ET trace: {et_file}")
    print(f"[Rank {rank}/{world_size}] Kineto trace: {kineto_file}")


if __name__ == "__main__":
    main()

# Example command (inference, single GPU):
#   python trace_gen/trace_gen.py --workload-card trace_collection/llama-3.1-8b-1GPU-DP-perlmutter/llama-3.1-8b-2GPU-pure-DP-perlmutter-inference.yaml
#
# Multi-GPU with torchrun (each rank writes its own trace):
#   torchrun --nproc_per_node=4 trace_gen/trace_gen.py --workload-card trace_collection/llama-3.1-8b-1GPU-DP-perlmutter/llama-3.1-8b-2GPU-pure-DP-perlmutter-inference.yaml
