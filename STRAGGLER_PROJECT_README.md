# Straggler Effect Analysis on 4 GPUs - Project Overview

This project provides comprehensive infrastructure to analyze **straggler effects** across different models and parallelization strategies in a scale-up (4 GPU) environment.

## Quick Start

1. **Read the comprehensive guide**: [EXECUTION_GUIDE.md](file:///global/homes/k/kg597/.gemini/antigravity/brain/e6678fc8-68c5-43d1-8ac9-21484172ac60/EXECUTION_GUIDE.md)
2. **Review the implementation plan**: [implementation_plan.md](file:///global/homes/k/kg597/.gemini/antigravity/brain/e6678fc8-68c5-43d1-8ac9-21484172ac60/implementation_plan.md)

## What's Been Created

### Documentation
- **EXECUTION_GUIDE.md** - Complete step-by-step guide for running all experiments
- **implementation_plan.md** - Technical plan with model/framework breakdown
- **task.md** - Task checklist tracking

### Scripts

#### Inference (vLLM + Tensor Parallelism)
- `trace_collection/scripts/inference/vllm_inference_tp.py` - vLLM inference with profiling
- `trace_collection/scripts/inference/run_vllm_inference.sh` - Wrapper script

#### Fine-tuning (TorchTitan + FSDP)
- `trace_collection/scripts/finetuning/torchtitan_train.py` - TorchTitan training with FSDP
- `trace_collection/scripts/finetuning/run_torchtitan.sh` - Wrapper script

#### Fine-tuning (DeepSpeed + ZeRO)
- `trace_collection/scripts/finetuning/deepspeed_train.py` - DeepSpeed training with ZeRO-2/3
- `trace_collection/scripts/finetuning/run_deepspeed.sh` - Wrapper script

#### Profiling & Analysis
- `trace_collection/scripts/profiling/nsys_profile_distributed.sh` - NSYS profiling wrapper
- `trace_collection/scripts/profiling/collect_all_traces.sh` - Master trace collection script
- `scripts/analyze_straggler.sh` - Batch analysis of straggler metrics

### Workload Cards (Templates)
- `trace_collection/workload_cards/llama-3.1-8b-vllm-tp4-4gpu.yaml`
- `trace_collection/workload_cards/llama-3.1-8b-torchtitan-fsdp-4gpu.yaml`
- `trace_collection/workload_cards/llama-3.1-8b-deepspeed-zero2-4gpu.yaml`

### Tools Integration
- Updated `tools/main.py` to support `straggler_delay` and `straggler_slowdown` metrics

## Experiment Coverage

### Models
- **Llama 3.1 8B** (recommended starting point)
- **DeepSeek V2 Lite** (MoE architecture)
- **Qwen 32B** (larger model for testing)

### Parallelization Strategies
- **Tensor Parallelism (TP)**: TP=2, TP=4
- **FSDP**: Full sharding across 4 GPUs
- **DeepSpeed ZeRO**: ZeRO-2 and ZeRO-3

### Workload Types
- **Inference**: Using vLLM with Tensor Parallelism
- **Fine-tuning**: Using TorchTitan (FSDP) or DeepSpeed (ZeRO)

## Running Experiments

### Environment Setup
```bash
cd ~/ccl-bench-straggler-scale-up
conda create --name ccl-bench python=3.10 -y
conda activate ccl-bench
pip install -r requirements.txt
pip install vllm transformers deepspeed torch
```

### Example: Inference Experiment
```bash
cd trace_collection/scripts/inference
./run_vllm_inference.sh "meta-llama/Llama-3.1-8B" 4 "llama-3.1-8b-vllm-tp4-4gpu"
```

### Example: Fine-tuning Experiment
```bash
cd trace_collection/scripts/finetuning
./run_torchtitan.sh "meta-llama/Llama-3.1-8B" "fsdp" "llama-3.1-8b-torchtitan-fsdp-4gpu"
```

### Example: Straggler Analysis
```bash
cd ~/ccl-bench-straggler-scale-up
python tools/main.py --trace ./trace_collection/llama-3.1-8b-vllm-tp4-4gpu --metric straggler_delay
python tools/main.py --trace ./trace_collection/llama-3.1-8b-vllm-tp4-4gpu --metric straggler_slowdown
```

### Batch Analysis
```bash
cd scripts
./analyze_straggler.sh ../trace_collection straggler_results.csv
```

## Expected Traces Per Experiment

After running an experiment, you should have:
```
trace_collection/<experiment-name>/
├── kineto_trace_0.json    # GPU 0 Kineto trace
├── kineto_trace_1.json    # GPU 1 Kineto trace
├── kineto_trace_2.json    # GPU 2 Kineto trace
├── kineto_trace_3.json    # GPU 3 Kineto trace
├── torch_et_0.json        # GPU 0 PyTorch ET
├── torch_et_1.json        # GPU 1 PyTorch ET
├── torch_et_2.json        # GPU 2 PyTorch ET
└── torch_et_3.json        # GPU 3 PyTorch ET
```

Optional NSYS traces:
```
├── nsys_0.nsys-rep
├── nsys_1.nsys-rep
├── nsys_2.nsys-rep
└── nsys_3.nsys-rep
```

## Straggler Metrics

Two metrics are implemented:

1. **Straggler Delay** - Normalized relative lag of slowest GPU
   - Range: [0, 1]
   - Lower is better (0 = perfect sync)

2. **Straggler Slowdown** - Ratio of slowest to fastest communication
   - Range: [1, ∞)
   - Lower is better (1 = perfect balance)

## Directory Structure

```
ccl-bench-straggler-scale-up/
├── README.md                           # Original repo README
├── requirements.txt
├── workload_card_template.yaml
├── trace_collection/
│   ├── scripts/
│   │   ├── inference/                 # vLLM inference scripts
│   │   │   ├── vllm_inference_tp.py
│   │   │   └── run_vllm_inference.sh
│   │   ├── finetuning/                # Training scripts
│   │   │   ├── torchtitan_train.py
│   │   │   ├── run_torchtitan.sh
│   │   │   ├── deepspeed_train.py
│   │   │   └── run_deepspeed.sh
│   │   └── profiling/                 # Profiling utilities
│   │       ├── nsys_profile_distributed.sh
│   │       └── collect_all_traces.sh
│   ├── workload_cards/                # Sample workload cards
│   │   ├── llama-3.1-8b-vllm-tp4-4gpu.yaml
│   │   ├── llama-3.1-8b-torchtitan-fsdp-4gpu.yaml
│   │   └── llama-3.1-8b-deepspeed-zero2-4gpu.yaml
│   └── <experiment-name>/             # Trace directories (created by scripts)
├── tools/
│   ├── main.py                        # Updated with straggler metrics
│   ├── straggler/
│   │   ├── straggler_delay.py
│   │   └── straggler_slowdown.py
│   └── coll_call_num/
└── scripts/
    └── analyze_straggler.sh           # Batch analysis script
```

## Next Steps

1. **Setup Environment**: Follow [EXECUTION_GUIDE.md](file:///global/homes/k/kg597/.gemini/antigravity/brain/e6678fc8-68c5-43d1-8ac9-21484172ac60/EXECUTION_GUIDE.md) Section 2
2. **Download Models**: Follow Section 3
3. **Run Experiments**: 
   - Start with inference experiments (Section 4)
   - Then fine-tuning experiments (Sections 5-6)
4. **Collect NSYS Traces** (optional): Section 7
5. **Analyze Results**: Section 8

## Documentation

All documentation is in the artifacts directory:
- **EXECUTION_GUIDE.md** - Full execution guide with all commands
- **implementation_plan.md** - Technical implementation details
- **task.md** - Task tracking checklist

## Troubleshooting

See [EXECUTION_GUIDE.md Section 10](file:///global/homes/k/kg597/.gemini/antigravity/brain/e6678fc8-68c5-43d1-8ac9-21484172ac60/EXECUTION_GUIDE.md#10-troubleshooting) for common issues and solutions.

## Notes

- All scripts are executable and ready to use
- Scripts include integrated PyTorch profiling (Kineto + ExecutionTraceObserver)
- NSYS profiling can be added separately for kernel-level analysis
- Workload cards are templates - adjust hardware specs to match your system
- All experiments are configured for 4 GPUs

---

**For detailed instructions and complete commands, refer to [EXECUTION_GUIDE.md](file:///global/homes/k/kg597/.gemini/antigravity/brain/e6678fc8-68c5-43d1-8ac9-21484172ac60/EXECUTION_GUIDE.md)**
