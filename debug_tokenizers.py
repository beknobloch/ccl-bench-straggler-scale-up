import os
from transformers import AutoTokenizer, LlamaTokenizer

mistral_path = "/pscratch/sd/k/kg597/llms/mistral-7b-instruct-v0.2"
qwen_path = "/pscratch/sd/k/kg597/llms/qwen3-vl-8b-instruct"

print(f"Testing Mistral from {mistral_path}")
try:
    print("Attempting AutoTokenizer (use_fast=True)...")
    tokenizer = AutoTokenizer.from_pretrained(mistral_path, use_fast=True)
    print("Success!")
    print(tokenizer.encode("Hello world"))
except Exception as e:
    print(f"Failed: {e}")

try:
    print("Attempting AutoTokenizer (use_fast=False)...")
    tokenizer = AutoTokenizer.from_pretrained(mistral_path, use_fast=False)
    print("Success!")
    print(tokenizer.encode("Hello world"))
except Exception as e:
    print(f"Failed: {e}")

try:
    print("Attempting LlamaTokenizer...")
    tokenizer = LlamaTokenizer.from_pretrained(mistral_path)
    print("Success!")
    print(tokenizer.encode("Hello world"))
except Exception as e:
    print(f"Failed: {e}")

print(f"\nTesting Qwen from {qwen_path}")
try:
    print("Attempting AutoTokenizer (use_fast=True)...")
    tokenizer = AutoTokenizer.from_pretrained(qwen_path, use_fast=True, trust_remote_code=True)
    print("Success!")
    print(tokenizer.encode("Hello world"))
except Exception as e:
    print(f"Failed: {e}")
