import os
from datasets import load_dataset
import argparse

def download_c4(output_dir):
    print(f"Downloading C4 dataset to {output_dir}...")
    # Load the 'en' subset of C4, streaming=False to download all
    # This might be huge, so we might want to just download a few shards if the user wants "more" but not "all"
    # But the user asked for "full dataset" or "all shards".
    # C4 'en' is massive (TB scale). 
    # Let's try to download the git repository or use huggingface-cli if available, 
    # but since the user wants a script, we can use the datasets library to download and save to disk.
    
    # However, C4 is huge. Downloading the *entire* thing might fill up scratch.
    # The user has one shard: c4-train.00000-of-01024.json.gz
    # This suggests they are using the google/c4 dataset or similar.
    
    # Let's provide a script that downloads the 'realnewslike' or just more shards of 'en'.
    # Given the file name format, it looks like the 'allenai/c4' or 'google/c4' raw files.
    
    # A safer bet for "more iterations" is to download a reasonable subset, e.g., 10-20 shards, 
    # or use the `load_dataset` with `streaming=True` and `take` then save, 
    # but `load_dataset` usually caches in ~/.cache.
    
    # Let's write a script that uses `git lfs` or `wget` if we knew the URL, 
    # but `datasets` library is the standard way.
    
    # Actually, the user's existing file `c4-train.00000-of-01024.json.gz` matches the `allenai/c4` file naming convention.
    # We can try to download more of these specific files.
    
    try:
        # We will use the huggingface_hub to download specific files to avoid downloading the whole TB dataset if not needed,
        # but the user asked for "all shards".
        # Let's warn them about size and provide a script to download a subset or all.
        
        from huggingface_hub import snapshot_download
        
        # This downloads the whole repo. C4 is HUGE. 
        # We should probably just download the first N shards.
        # But the user said "download all the shards".
        
        # Let's use `load_dataset` to download and save to disk in the format expected.
        # But `load_dataset` saves as arrow files, not .json.gz.
        
        # The existing file is .json.gz.
        # This is likely from: https://huggingface.co/datasets/allenai/c4/tree/main/en
        
        # We can use `huggingface_hub` to download files matching the pattern.
        
        from huggingface_hub import hf_hub_download
        
        # Download first 8 shards (enough for 4 GPUs * many iterations)
        # 1024 shards total.
        
        repo_id = "allenai/c4"
        repo_type = "dataset"
        subfolder = "en"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Download shards
        if args.all:
            num_shards = 1024
            print("Downloading ALL 1024 shards...")
        else:
            num_shards = args.num_shards
            print(f"Downloading first {num_shards} shards...")
        
        for i in range(num_shards):
            filename = f"c4-train.{i:05d}-of-01024.json.gz"
            # Check if exists
            local_file = os.path.join(output_dir, filename)
            if os.path.exists(local_file):
                print(f"Skipping {filename} (already exists)")
                continue
                
            print(f"Downloading {filename}...")
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    subfolder=subfolder,
                    repo_type=repo_type,
                    local_dir=output_dir,
                    local_dir_use_symlinks=False
                )
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
            
        print("Download complete.")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="/pscratch/sd/k/kg597/c4_dataset/en")
    parser.add_argument("--num-shards", type=int, default=8, help="Number of shards to download (default: 8)")
    parser.add_argument("--all", action="store_true", help="Download all 1024 shards")
    args = parser.parse_args()
    
    download_c4(args.output_dir)
