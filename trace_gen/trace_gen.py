import torch
import yaml
import os
import argparse
import json
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoModelForCausalLM, AutoTokenizer
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

# Load tokenizer that matches the model
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
# Ensure we have a pad token for batching
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

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
    
    # Tensor Parallelism: shard model layers across GPUs
    if parallel_dims.tp_enabled:
        if rank == 0:
            print("Applying Tensor Parallelism...")
        tp_mesh = world_mesh["tp"]
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
        
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        from functools import partial
        
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
            limit_all_gathers=True,
            forward_prefetch=True,
        )
    else:
        model = model.to(device)
else:
    model = model.to(device)

# -----------------------------
# Load C4 dataset shard instead of dummy data
# -----------------------------
def load_c4_texts(path, max_docs=None):
    texts = []
    with open(path, "r") as f:
        # Try to detect if file is JSON lines or a JSON array
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            data = json.load(f)
            for d in data:
                t = d.get("text") or d.get("content") or d.get("raw")
                if t:
                    texts.append(t)
                    if max_docs and len(texts) >= max_docs:
                        break
        else:
            # JSON Lines
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                t = obj.get("text") or obj.get("content") or obj.get("raw")
                if t:
                    texts.append(t)
                    if max_docs and len(texts) >= max_docs:
                        break
    return texts

if rank == 0:
    print("Loading C4 shard...")

c4_path = "c4_dataset/en/c4-train.00000-of-01024.json"

# Load enough docs for all ranks
max_docs = batch_size * world_size
all_texts = load_c4_texts(c4_path, max_docs=max_docs)

if len(all_texts) == 0:
    raise RuntimeError(f"No usable texts found in {c4_path}")

# Give each rank its own slice of texts
rank_texts = all_texts[rank::world_size]
if len(rank_texts) < batch_size:
    # Repeat texts if there are not enough for this rank
    repeat_factor = (batch_size + len(rank_texts) - 1) // len(rank_texts)
    rank_texts = (rank_texts * repeat_factor)[:batch_size]
else:
    rank_texts = rank_texts[:batch_size]

if rank == 0:
    print("Example input text:", rank_texts[0][:200])

enc = tokenizer(
    rank_texts,
    padding="max_length",
    truncation=True,
    max_length=seq_len,
    return_tensors="pt",
)

inputs = enc["input_ids"].to(device)
attention_mask = enc["attention_mask"].to(device)

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
        # Forward-only (inference) mode
        with record_function("forward_only"):
            with torch.no_grad():
                output = model(inputs, attention_mask=attention_mask)
            
            if world_size > 1:
                dist.barrier()
    else:
        # Forward + Backward (training) mode
        with record_function("forward_backward"):
            output = model(inputs, attention_mask=attention_mask)
            logits = output.logits
            loss = logits.sum()
            loss.backward()
            
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

# salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 4 --account m4999
# hf auth login
# export HF_HOME=/pscratch/sd/m/mh2653/.cache/huggingface

# torchrun --nproc_per_node=4 trace_gen/trace_gen.py -- --workload_card trace_collection/llama-3.1-8b-1GPU-DP-perlmutter/llama-3.1-8b-2GPU-pure-DP-perlmutter-inference.yaml

# python tools/main.py --trace ./ --metric coll_call_num
# python tools/main.py --trace ./ --metric straggler_delay
# python tools/main.py --trace ./ --metric straggler_slowdown

# FSDP (2GPUs)
# coll_call_num: rank0: 99, rank1: 99
# straggler_delay: 3.095612673844695e-07
# straggler_slowdown: 1.0000390489546451

# FSDP (4GPUs)
# coll_call_num: rank0: 99, rank1: 99, rank2: 99, rank3: 99
# straggler_delay: 5.303492827743813e-07
# straggler_slowdown: 1.0007944156004465

# TP (2GPUs)
# coll_call_num: rank0: 129, rank1: 129
# straggler_delay: 1.129688648118878e-06
# straggler_slowdown: 1.0000241782508983

# TP (4GPUs)
# coll_call_num: rank0: 129, rank1: 129, rank2: 129, rank3: 129
# straggler_delay: 4.903407062305596e-07
# straggler_slowdown: 1.0002771298074606