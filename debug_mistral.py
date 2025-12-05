from transformers import AutoTokenizer

model_path = "/pscratch/sd/k/kg597/llms/mistral-7b-instruct-v0.2"
print(f"Loading tokenizer from {model_path}")

try:
    # Try default loading
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    text = ["Test sentence for tokenizer."]
    print("Tokenizing...")
    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    print("Success with use_fast=True!")
except Exception as e:
    print(f"Failed with use_fast=True: {e}")

print("-" * 20)

try:
    # Try slow tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    text = ["Test sentence for tokenizer."]
    print("Tokenizing...")
    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    print("Success with use_fast=False!")
except Exception as e:
    print(f"Failed with use_fast=False: {e}")
