# Model Path Configuration for PSCRATCH
# Use these local paths instead of downloading from HuggingFace

# Available models in $PSCRATCH
export PSCRATCH="/pscratch/sd/k/kg597"

# Llama 3.1 8B (use HuggingFace cache - has all required files)
export LLAMA_3_1_8B="/pscratch/sd/k/kg597/llms/llama-3.1-8b-instruct"

# Alternative: Llama 3.1 8B Instruct (missing config.json - don't use for vLLM)
# export LLAMA_3_1_8B_INSTRUCT="$PSCRATCH/llms/llama-3.1-8b-instruct"

# Mistral 7B Instruct
export MISTRAL_7B="$PSCRATCH/llms/mistral-7b-instruct-v0.2"

# Qwen3 VL 8B Instruct
export QWEN3_VL_8B="$PSCRATCH/llms/qwen3-vl-8b-instruct"

# Alternative: HuggingFace cache locations
export LLAMA_3_1_8B_HF="$PSCRATCH/huggingface/hub/models--meta-llama--Llama-3.1-8B"

# Usage in scripts:
# Instead of: "meta-llama/Llama-3.1-8B"
# Use: "$LLAMA_3_1_8B" or "/pscratch/sd/k/kg597/llms/llama-3.1-8b-instruct"

echo "Model paths configured:"
echo "  LLAMA_3_1_8B: $LLAMA_3_1_8B"
echo "  MISTRAL_7B: $MISTRAL_7B"
echo "  QWEN3_VL_8B: $QWEN3_VL_8B"
