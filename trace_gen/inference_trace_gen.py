import torch
import yaml
import os
import argparse
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoModelForCausalLM
import torch.distributed as dist
from torchtitan.distributed import ParallelDims
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, CPUOffload
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
from torch.distributed.tensor.parallel.style import PrepareModuleInput

# -----------------------------
# Setup distributed environment
# -----------------------------
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
        return 0, 1, 0

# -----------------------------
# Load workload card parameters
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--workload_card", type=str, required=True)
parser.add_argument("--forward-only", action="store_true", help="Skip backward pass (forward inference only)")
parser.add_argument("--batch-size", type=int, default=None, help="Override batch size from workload card")
parser.add_argument("--seq-len", type=int, default=None, help="Override sequence length from workload card")
args = parser.parse_args()

with open(args.workload_card, "r") as f:
    card = yaml.safe_load(f)

# Extract model path from HF URL
hf_url = card["hf_url"]
model_name = '/'.join(hf_url.split('/')[-2:])

batch_size = args.batch_size if args.batch_size else card["workload"]["data"]["batch_size"]
seq_len = args.seq_len if args.seq_len else card["workload"]["data"]["seq_len"]

dp_replicate = card["Model-executor"]["model_plan_parallelization"].get("dp_replicate", 1)
dp_shard = card["Model-executor"]["model_plan_parallelization"].get("dp_shard", 1)
tp = card["Model-executor"]["model_plan_parallelization"]["tp"]
pp = card["Model-executor"]["model_plan_parallelization"]["pp"]
cp = card["Model-executor"]["model_plan_parallelization"].get("cp", 1)

# Initialize distributed
rank, world_size, local_rank = setup_distributed()
device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

if rank == 0:
    print(f"World size: {world_size}, Device: {device}")
    print(f"Model: {model_name}, Batch size: {batch_size}, Seq length: {seq_len}")
    print(f"DP replicate: {dp_replicate}, DP shard: {dp_shard}, TP: {tp}, PP: {pp}, CP: {cp}")

# -----------------------------
# Build TorchTitan parallel dims
# -----------------------------
parallel_dims = ParallelDims(
    dp_replicate=dp_replicate,
    dp_shard=dp_shard,
    tp=tp,
    pp=pp,
    cp=cp,
    ep=1,
    etp=1,
    world_size=world_size,
)

# -----------------------------
# Load model with HuggingFace
# -----------------------------
if rank == 0:
    print("Loading model...")

# Load on CPU to avoid OOM during initialization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=None,
)

# Enable gradient checkpointing BEFORE FSDP wrapping
if hasattr(model, 'gradient_checkpointing_enable'):
    model.gradient_checkpointing_enable()
    if rank == 0:
        print("Gradient checkpointing enabled")

# Apply parallelism using TorchTitan's ParallelDims
if world_size > 1:
    world_mesh = parallel_dims.build_mesh()
    if rank == 0:
        print(f"Built device mesh: {world_mesh}")
    
    # Apply different parallelism strategies based on config
    if parallel_dims.tp_enabled:
        # Tensor Parallelism: shard model layers across GPUs
        if rank == 0:
            print("Applying Tensor Parallelism...")
        tp_mesh = world_mesh["tp"]
        # Apply TP to attention layers (example for LLaMA)
        plan = {
            "model.layers.*.self_attn.q_proj": ColwiseParallel(),
            "model.layers.*.self_attn.k_proj": ColwiseParallel(),
            "model.layers.*.self_attn.v_proj": ColwiseParallel(),
            "model.layers.*.self_attn.o_proj": RowwiseParallel(),
        }
        model = parallelize_module(model, tp_mesh, plan)
    
    if parallel_dims.dp_shard_enabled:
        # FSDP: shard parameters and gradients
        if rank == 0:
            print("Applying FSDP with per-layer wrapping...")
        
        # Import the transformer layer class for wrapping
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        from functools import partial
        
        # Create auto-wrap policy to wrap each transformer layer
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={LlamaDecoderLayer},
        )
        
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            use_orig_params=True,
            limit_all_gathers=True,  # Reduce memory for all-gather
            forward_prefetch=True,   # Prefetch next layer
        )
    else:
        model = model.to(device)
else:
    model = model.to(device)

# -----------------------------
# Generate dummy data
# -----------------------------
inputs = torch.randint(
    low=1,
    high=32000,
    size=(batch_size, seq_len),
    device=device
)

# -----------------------------
# Run trace (1 iteration)
# -----------------------------
if rank == 0:
    print("Running trace...")

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True,
    profile_memory=True,
    with_modules=True,
) as prof:

    if args.forward_only:
        # Forward-only (inference) mode - saves memory
        with record_function("forward_only"):
            with torch.no_grad():
                output = model(inputs)
            
            # Sync barrier for multi-GPU
            if world_size > 1:
                dist.barrier()
    else:
        # Forward + Backward (training) mode
        with record_function("forward_backward"):
            output = model(inputs)
            # Get logits from the model output and compute a simple loss
            logits = output.logits
            loss = logits.sum()  # Simple loss for tracing
            loss.backward()
            
            # Sync barrier for multi-GPU
            if world_size > 1:
                dist.barrier()

# -----------------------------
# Save trace
# -----------------------------
trace_file = f"trace_rank_{rank}.json"
prof.export_chrome_trace(trace_file)
print(f"Rank {rank}: Trace saved to {trace_file}")

# Cleanup
if world_size > 1:
    dist.destroy_process_group()
