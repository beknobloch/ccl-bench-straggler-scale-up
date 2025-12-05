import torch
import yaml
import os
import argparse
import json
import glob
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.distributed as dist
from tqdm import tqdm
from torchtitan.distributed import ParallelDims
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, CPUOffload
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
from torch.distributed.tensor.parallel.style import PrepareModuleInput
import gzip

# -----------------------------
# Monkeypatch Transformers for Mistral
# -----------------------------
try:
    from transformers import tokenization_mistral_common
    from transformers.tokenization_mistral_common import MistralTokenizerType, MistralCommonTokenizer

    original_is_control_token = MistralCommonTokenizer._is_control_token

    def _is_control_token_fixed(self, token_id: int) -> bool:
        if self._tokenizer_type == MistralTokenizerType.spm:
            # Fix: Handle _control_tokens being a set (not callable)
            ct = self.tokenizer.instruct_tokenizer.tokenizer._control_tokens
            if callable(ct):
                return token_id in ct()
            return token_id in ct
        elif self._tokenizer_type == MistralTokenizerType.tekken:
            return token_id < self.tokenizer.instruct_tokenizer.tokenizer.num_special_tokens
        else:
            raise ValueError(f"Unknown tokenizer type: {self._tokenizer_type}")

    MistralCommonTokenizer._is_control_token = _is_control_token_fixed
    print("Applied monkeypatch for MistralCommonTokenizer._is_control_token")
except ImportError:
    pass

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
parser.add_argument("--max-docs", type=int, default=None, help="Optional limit on number of docs to load from dataset")
parser.add_argument("--output-dir", type=str, default=None, help="Directory to save traces")
args = parser.parse_args()

with open(args.workload_card, "r") as f:
    card = yaml.safe_load(f)

# Extract model path from HF URL
hf_url = card["hf_url"]
if hf_url.startswith("/"):
    model_name = hf_url
else:
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
# Load tokenizer that matches the model
if "mistral" in model_name.lower():
    from transformers import LlamaTokenizerFast
    if rank == 0:
        print("Using LlamaTokenizerFast for Mistral model to avoid mistral-common bugs.")
    tokenizer = LlamaTokenizerFast.from_pretrained(model_name)
else:
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
        
        # Determine architecture for TP plan
        # Llama
        if "llama" in model_name.lower():
            plan = {
                "model.layers.*.self_attn.q_proj": ColwiseParallel(),
                "model.layers.*.self_attn.k_proj": ColwiseParallel(),
                "model.layers.*.self_attn.v_proj": ColwiseParallel(),
                "model.layers.*.self_attn.o_proj": RowwiseParallel(),
                "model.layers.*.mlp.gate_proj": ColwiseParallel(),
                "model.layers.*.mlp.up_proj": ColwiseParallel(),
                "model.layers.*.mlp.down_proj": RowwiseParallel(),
            }
        # Mistral (similar to Llama)
        elif "mistral" in model_name.lower():
            plan = {
                "model.layers.*.self_attn.q_proj": ColwiseParallel(),
                "model.layers.*.self_attn.k_proj": ColwiseParallel(),
                "model.layers.*.self_attn.v_proj": ColwiseParallel(),
                "model.layers.*.self_attn.o_proj": RowwiseParallel(),
                "model.layers.*.mlp.gate_proj": ColwiseParallel(),
                "model.layers.*.mlp.up_proj": ColwiseParallel(),
                "model.layers.*.mlp.down_proj": RowwiseParallel(),
            }
        # Qwen (verify layer names, usually similar to Llama)
        elif "qwen" in model_name.lower():
             plan = {
                "model.layers.*.self_attn.q_proj": ColwiseParallel(),
                "model.layers.*.self_attn.k_proj": ColwiseParallel(),
                "model.layers.*.self_attn.v_proj": ColwiseParallel(),
                "model.layers.*.self_attn.o_proj": RowwiseParallel(),
                "model.layers.*.mlp.gate_proj": ColwiseParallel(),
                "model.layers.*.mlp.up_proj": ColwiseParallel(),
                "model.layers.*.mlp.down_proj": RowwiseParallel(),
            }
        else:
            # Fallback for Llama-like
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
        
        # Import correct decoder layer based on model
        try:
            if "mistral" in model_name.lower():
                from transformers.models.mistral.modeling_mistral import MistralDecoderLayer as DecoderLayer
            elif "qwen" in model_name.lower():
                from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer as DecoderLayer
            else:
                from transformers.models.llama.modeling_llama import LlamaDecoderLayer as DecoderLayer
        except ImportError:
             # Fallback
             from transformers.models.llama.modeling_llama import LlamaDecoderLayer as DecoderLayer

        from functools import partial
        
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={DecoderLayer},
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
    open_func = gzip.open if path.endswith(".gz") else open
    
    with open_func(path, "rt", encoding="utf-8") as f:
        # Try to detect if file is JSON lines or a JSON array
        # Peeking with gzip stream is tricky, let's assume JSONL for C4 usually
        # But let's try to be robust
        try:
            first_char = f.read(1)
        except Exception:
            # Empty file or read error
            return []
            
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
                try:
                    obj = json.loads(line)
                    t = obj.get("text") or obj.get("content") or obj.get("raw")
                    if t:
                        texts.append(t)
                        if max_docs and len(texts) >= max_docs:
                            break
                except json.JSONDecodeError:
                    continue
    return texts

if rank == 0:
    print("Loading C4 shard...")

c4_dir = "/pscratch/sd/k/kg597/c4_dataset/en"
# Check for nested 'en' directory where huggingface_hub might have put the files
nested_c4_dir = os.path.join(c4_dir, "en")
if os.path.isdir(nested_c4_dir):
    c4_files = sorted(glob.glob(os.path.join(nested_c4_dir, "c4-train.*.json*")))
    if not c4_files:
        # Fallback to parent dir
        c4_files = sorted(glob.glob(os.path.join(c4_dir, "c4-train.*.json*")))
else:
    c4_files = sorted(glob.glob(os.path.join(c4_dir, "c4-train.*.json*")))

if not c4_files:
    # Fallback to specific path if glob fails or dir doesn't exist
    c4_files = ["/pscratch/sd/k/kg597/c4_dataset/en/c4-train.00000-of-01024.json"]

all_texts = []
for c4_path in c4_files:
    if args.max_docs and len(all_texts) >= args.max_docs:
        break
    
    if rank == 0:
        print(f"Loading {c4_path}...")
    
    try:
        new_texts = load_c4_texts(c4_path, max_docs=args.max_docs - len(all_texts) if args.max_docs else None)
        all_texts.extend(new_texts)
    except Exception as e:
        if rank == 0:
            print(f"Skipping {c4_path} due to error: {e}")
            
if rank == 0:
    print(f"Total files loaded: {len(c4_files)}")

if len(all_texts) == 0:
    raise RuntimeError(f"No usable texts found in {c4_dir}")

# Give each rank its own slice of texts
rank_texts = all_texts[rank::world_size]
if len(rank_texts) == 0:
    raise RuntimeError(f"Rank {rank} has no assigned texts from dataset slice")

if rank == 0:
    print(f"Total docs loaded: {len(all_texts)}, docs per rank: {len(rank_texts)}")
    if rank_texts:
        print("Example input text:", rank_texts[0][:200])
num_batches = (len(rank_texts) + batch_size - 1) // batch_size

# -----------------------------
# Run trace (iterate over dataset)
# -----------------------------
if rank == 0:
    print("Running trace...")

progress_iter = tqdm(range(num_batches), desc="Batches", disable=rank != 0)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True,
    profile_memory=True,
    with_modules=True,
) as prof:

    for batch_idx in progress_iter:
        start = batch_idx * batch_size
        end = min(start + batch_size, len(rank_texts))
        batch_texts = rank_texts[start:end]

        enc = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=seq_len,
            return_tensors="pt",
        )

        inputs = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # Inference mode (forward only)
        with record_function(f"inference_batch_{batch_idx}"):
            with torch.no_grad():
                output = model(inputs, attention_mask=attention_mask)
            
            if world_size > 1:
                dist.barrier()

# -----------------------------
# Save trace
# -----------------------------
output_dir = args.output_dir if args.output_dir else "."
if rank == 0:
    os.makedirs(output_dir, exist_ok=True)
if world_size > 1:
    dist.barrier()

trace_file = os.path.join(output_dir, f"trace_rank_{rank}.json")
prof.export_chrome_trace(trace_file)
print(f"Rank {rank}: Trace saved to {trace_file}")

# Cleanup
if world_size > 1:
    dist.destroy_process_group()
