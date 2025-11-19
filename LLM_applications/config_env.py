import logging
logging.getLogger().setLevel(logging.ERROR)
import os

cache_root = "huggingface_path"

os.environ["HF_HOME"] = cache_root  
import transformers
import transformer_lens
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer, ActivationCache