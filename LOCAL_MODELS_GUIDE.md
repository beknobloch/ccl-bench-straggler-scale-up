# Using Local Models from $PSCRATCH

## Available Models

You have the following models already downloaded in `$PSCRATCH`:

### In `/pscratch/sd/k/kg597/llms/`:
- **Llama 3.1 8B Instruct**: `/pscratch/sd/k/kg597/llms/llama-3.1-8b-instruct`
- **Mistral 7B Instruct v0.2**: `/pscratch/sd/k/kg597/llms/mistral-7b-instruct-v0.2`
- **Qwen3 VL 8B Instruct**: `/pscratch/sd/k/kg597/llms/qwen3-vl-8b-instruct`

### In HuggingFace cache (`/pscratch/sd/k/kg597/huggingface/hub/`):
- **Llama 3.1 8B**: `/pscratch/sd/k/kg597/huggingface/hub/models--meta-llama--Llama-3.1-8B`
- **CodeLlama 34B**: `/pscratch/sd/k/kg597/huggingface/hub/models--meta-llama--CodeLlama-34b-hf`

## Quick Setup

Source the model paths configuration:
```bash
cd ~/ccl-bench-straggler-scale-up
source model_paths.sh
```

## Usage in Scripts

### Option 1: Use Environment Variables (Recommended)

```bash
# Source the config first
source model_paths.sh

# Then use the variables
./run_vllm_inference.sh "$LLAMA_3_1_8B" 4 "llama-3.1-8b-vllm-tp4-4gpu"
./run_torchtitan.sh "$LLAMA_3_1_8B" "fsdp" "llama-3.1-8b-torchtitan-fsdp-4gpu"
./run_deepspeed.sh "$LLAMA_3_1_8B" 2 "llama-3.1-8b-deepspeed-zero2-4gpu"
```

### Option 2: Use Full Paths Directly

```bash
# Inference
cd trace_collection/scripts/inference
./run_vllm_inference.sh \
    "/pscratch/sd/k/kg597/llms/llama-3.1-8b-instruct" \
    4 \
    "llama-3.1-8b-vllm-tp4-4gpu"

# TorchTitan
cd ../finetuning
./run_torchtitan.sh \
    "/pscratch/sd/k/kg597/llms/llama-3.1-8b-instruct" \
    "fsdp" \
    "llama-3.1-8b-torchtitan-fsdp-4gpu"

# DeepSpeed
./run_deepspeed.sh \
    "/pscratch/sd/k/kg597/llms/llama-3.1-8b-instruct" \
    2 \
    "llama-3.1-8b-deepspeed-zero2-4gpu"
```

## Missing Models

You currently **do not have** the following models mentioned in the execution guide:
- DeepSeek V2 Lite
- Qwen 32B (you have Qwen3 VL 8B instead)

### Options:
1. **Use what you have**: Run experiments with Llama 3.1 8B, Mistral 7B, and Qwen3 VL 8B
2. **Download missing models** to `$PSCRATCH/llms/`:
   ```bash
   cd $PSCRATCH/llms
   huggingface-cli download deepseek-ai/DeepSeek-V2-Lite --local-dir deepseek-v2-lite
   huggingface-cli download Qwen/Qwen2-32B --local-dir qwen2-32b
   ```

## Recommended Experiments with Your Models

### Inference (vLLM)
```bash
source model_paths.sh
cd trace_collection/scripts/inference

# Llama 3.1 8B with TP=4
./run_vllm_inference.sh "$LLAMA_3_1_8B" 4 "llama-3.1-8b-vllm-tp4-4gpu"

# Llama 3.1 8B with TP=2
./run_vllm_inference.sh "$LLAMA_3_1_8B" 2 "llama-3.1-8b-vllm-tp2-4gpu"

# Mistral 7B with TP=4
./run_vllm_inference.sh "$MISTRAL_7B" 4 "mistral-7b-vllm-tp4-4gpu"
```

### Fine-tuning (TorchTitan)
```bash
cd ../finetuning

# Llama 3.1 8B with FSDP
./run_torchtitan.sh "$LLAMA_3_1_8B" "fsdp" "llama-3.1-8b-torchtitan-fsdp-4gpu"

# Mistral 7B with FSDP
./run_torchtitan.sh "$MISTRAL_7B" "fsdp" "mistral-7b-torchtitan-fsdp-4gpu"
```

### Fine-tuning (DeepSpeed)
```bash
# Llama 3.1 8B with ZeRO-2
./run_deepspeed.sh "$LLAMA_3_1_8B" 2 "llama-3.1-8b-deepspeed-zero2-4gpu"

# Llama 3.1 8B with ZeRO-3
./run_deepspeed.sh "$LLAMA_3_1_8B" 3 "llama-3.1-8b-deepspeed-zero3-4gpu"

# Mistral 7B with ZeRO-2
./run_deepspeed.sh "$MISTRAL_7B" 2 "mistral-7b-deepspeed-zero2-4gpu"
```

## Verification

Verify your model paths work:
```bash
# Check Llama 3.1 8B
ls -lh /pscratch/sd/k/kg597/llms/llama-3.1-8b-instruct/

# Check Mistral 7B
ls -lh /pscratch/sd/k/kg597/llms/mistral-7b-instruct-v0.2/

# Check Qwen3 VL 8B
ls -lh /pscratch/sd/k/kg597/llms/qwen3-vl-8b-instruct/
```

All should show model files (`.safetensors`, `tokenizer.json`, `config.json`, etc.)
