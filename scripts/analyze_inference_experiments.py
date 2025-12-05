import os
import subprocess
import argparse
import glob

def analyze_inference_traces():
    base_trace_dir = "/pscratch/sd/k/kg597/traces/inference"
    
    # Find all subdirectories in the inference trace folder
    if not os.path.exists(base_trace_dir):
        print(f"No inference traces found at {base_trace_dir}")
        return

    experiments = sorted([d for d in os.listdir(base_trace_dir) if os.path.isdir(os.path.join(base_trace_dir, d))])
    
    metrics = ["straggler_delay", "straggler_slowdown"]
    
    print(f"{'Experiment':<40} | {'Metric':<20} | {'Value':<15}")
    print(f"{'-'*40}-|-{'-'*20}-|-{'-'*15}")
    
    for exp in experiments:
        trace_dir = os.path.join(base_trace_dir, exp)
            
        print(f"\nAnalyzing INFERENCE {exp}...")
        for metric in metrics:
            cmd = ["python", "tools/main.py", "--trace", trace_dir, "--metric", metric]
            
            try:
                print(f"--- {metric} ---")
                subprocess.run(cmd, check=True)
                
            except subprocess.CalledProcessError:
                print(f"Error calculating {metric} for {exp}")

if __name__ == "__main__":
    analyze_inference_traces()
