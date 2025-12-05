from transformers import AutoTokenizer, LlamaTokenizer, LlamaTokenizerFast
import os

model_path = "/pscratch/sd/k/kg597/llms/mistral-7b-instruct-v0.2"

print(f"Loading from {model_path}")

def test_tokenizer(name, tokenizer_cls, **kwargs):
    print(f"\n--- Testing {name} ---")
    try:
        tokenizer = tokenizer_cls.from_pretrained(model_path, **kwargs)
        print(f"Class: {type(tokenizer)}")
        print(f"pad_token_id: {tokenizer.pad_token_id}")
        print(f"eos_token_id: {tokenizer.eos_token_id}")
        print(f"vocab_size: {tokenizer.vocab_size}")
        
        # Try to access pad_token
        try:
            print(f"pad_token: {tokenizer.pad_token}")
        except Exception as e:
            print(f"Error accessing pad_token: {e}")
            
        # Try to set pad_token if None
        if tokenizer.pad_token_id is None:
            print("Setting pad_token = eos_token")
            tokenizer.pad_token = tokenizer.eos_token
            print(f"New pad_token_id: {tokenizer.pad_token_id}")
            
        # Try to tokenize
        enc = tokenizer("Test sentence", padding="max_length", max_length=10, return_tensors="pt")
        print("Tokenization successful")
        
    except Exception as e:
        print(f"Initialization failed: {e}")

test_tokenizer("AutoTokenizer (fast)", AutoTokenizer, use_fast=True)
test_tokenizer("LlamaTokenizerFast", LlamaTokenizerFast)
test_tokenizer("LlamaTokenizer (slow)", LlamaTokenizer)
