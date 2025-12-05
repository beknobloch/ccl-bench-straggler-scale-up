import os
import argparse
import subprocess
import glob

def run_experiments(args):
    workload_cards_dir = "trace_collection/workload_cards"
    # List all inference cards
    cards = sorted([os.path.basename(f) for f in glob.glob(os.path.join(workload_cards_dir, "*-inference.yaml"))])
    
    base_cmd = [
        "torchrun",
        "--nproc_per_node=4",
        "trace_gen/trace_gen_inference.py",
        "--max-docs", str(args.max_docs)
    ]
    
    # Ensure HF_HOME is set to the correct cache directory
    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = "/pscratch/sd/k/kg597/huggingface"
        print(f"Setting HF_HOME to {os.environ['HF_HOME']}")
    
    for card in cards:
        card_path = os.path.join(workload_cards_dir, card)
        if not os.path.exists(card_path):
            print(f"Warning: Card {card_path} not found. Skipping.")
            continue
            
        experiment_name = card.replace(".yaml", "")
        # Save to traces/inference/<experiment_name>
        output_dir = os.path.join("/pscratch/sd/k/kg597/traces/inference", experiment_name)
        os.makedirs(output_dir, exist_ok=True)
        
        cmd = base_cmd + ["--workload_card", card_path, "--output-dir", output_dir]
        
        print(f"\n{'='*60}")
        print(f"Running INFERENCE experiment with {card}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*60}\n")
        
        if not args.dry_run:
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running {card}: {e}")
                if not args.continue_on_error:
                    break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-docs", type=int, default=128, help="Max documents to process (controls iterations)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue to next experiment if one fails")
    args = parser.parse_args()
    
    run_experiments(args)
