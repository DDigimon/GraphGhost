import config_env

from config_env import AutoTokenizer, AutoModelForCausalLM, HookedTransformer
import torch


def model_name_func(name):
    if name == 'gemma-2B':
        return 'google/gemma-2-2b'
    if name == 'gemma-1B':
        return 'google/gemma-3-1b'
    if name == 'Qwen-0.5B':
        return "Qwen/Qwen2.5-0.5B"
    if name == 'Qwen3-0.6B':
        return "Qwen/Qwen3-0.6B"
    if name == 'Qwen3-4B':
        return "Qwen/Qwen3-4B"
    if name == 'Qwen-1.5B':
        return "Qwen/Qwen2.5-1.5B"
    if name == 'Llama-1B':
        return 'meta-llama/Llama-3.2-1B'
    if name == 'Llama-3B':
        return 'meta-llama/Llama-3.2-3B'
    if name == 'Llama-8B':
        return 'meta-llama/Llama-3.1-8B'
    if name == 'Llama-8BI':
        return 'meta-llama/Llama-3.1-8B-Instruct'
    if name == 'ds-qwen-1.5B':
        return 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
    if name == 'ds-llama':
        return 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'

def load_tokenizer(name):
    model_name = model_name_func(name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    return tokenizer

def load_model(name, device='cuda'):
    model_name = model_name_func(name)
    if 'ds' in name:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
        )
        
        if 'qwen-1.5B' in name:
            backbone_model = "Qwen/Qwen2.5-1.5B"    
        if 'llama' in name:
            backbone_model = "meta-llama/Llama-3.1-8B"    
        model = HookedTransformer.from_pretrained(
            backbone_model,
            hf_model=hf_model,
            tokenizer=tok,  
            device=device,
            dtype=torch.float16,
            trust_remote_code=True,
        )
        
    else:
        model = HookedTransformer.from_pretrained(
            model_name=model_name,
            device=device,
        )
        
    return model