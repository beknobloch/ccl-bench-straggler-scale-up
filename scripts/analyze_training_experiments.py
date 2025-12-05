import os
import subprocess
import argparse

def analyze_all_traces():
    base_trace_dir = "/pscratch/sd/k/kg597/traces"
    
    # List of all expected experiment directories
    experiments = [
        "llama-3.1-8b-4gpu-dp",
        "llama-3.1-8b-4gpu-fsdp",
        "llama-3.1-8b-4gpu-tp",
        "llama-3.1-8b-4gpu-tp2-dp2",
        "mistral-7b-4gpu-dp",
        "mistral-7b-4gpu-fsdp",
        "mistral-7b-4gpu-tp",
        "mistral-7b-4gpu-tp2-dp2",
        "qwen-8b-4gpu-dp",
        "qwen-8b-4gpu-fsdp",
        "qwen-8b-4gpu-tp",
        "qwen-8b-4gpu-tp2-dp2"
    ]
    
    metrics = ["straggler_delay", "straggler_slowdown"]
    
    print(f"{'Experiment':<30} | {'Metric':<20} | {'Value':<15}")
    print(f"{'-'*30}-|-{'-'*20}-|-{'-'*15}")
    
    for exp in experiments:
        trace_dir = os.path.join(base_trace_dir, exp)
        if not os.path.exists(trace_dir):
            print(f"{exp:<30} | {'ALL':<20} | {'Not Found':<15}")
            continue
            
        print(f"\nAnalyzing {exp}...")
        for metric in metrics:
            cmd = ["python", "tools/main.py", "--trace", trace_dir, "--metric", metric]
            
            try:
                # Run and capture output to print detailed stats if needed, or just print the final line
                # Since we modified the tools to print details, let's show them!
                print(f"--- {metric} ---")
                subprocess.run(cmd, check=True)
                
            except subprocess.CalledProcessError:
                print(f"Error calculating {metric} for {exp}")

if __name__ == "__main__":
    analyze_all_traces()
