
import os

from torch.utils.data import Dataset, DataLoader, IterableDataset
from tqdm import tqdm
import torch
from transformer_lens import HookedTransformerConfig
from transformer_lens import HookedTransformer


def build_cfg(c):
    return HookedTransformerConfig(
        n_layers=c["n_layers"],
        d_model=c["d_model"],
        d_head=c["d_model"] // c["n_head"],
        n_ctx=c["max_length"],
        d_vocab=c["vocab_size"],
        seed=c["seed"],
        act_fn=c["act_fn"],
        init_mode=c["init_mode"],
        positional_embedding_type=c["positions"]
    )


class MyDataset(Dataset):
    def __init__(self, texts, tokenizer, stack = True):
        self.examples = []
        for text in tqdm(texts, total=len(texts)):
            ids = tokenizer.encode(text, add_special_tokens=False)
            self.examples.append(torch.tensor(ids).long())
        if stack:
            self.examples = torch.cat(self.examples, 0)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
    
class MyTestDataset(Dataset):
    def __init__(self, texts, tokenizer, key_Word='START_A'):
        self.examples = []
        self.ans = []
        for text in texts:
            question, ans = text.split(key_Word)
            ids = tokenizer.encode(question, add_special_tokens=False)
            self.examples.append(torch.tensor(ids))
            self.ans.append(ans)
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx], self.ans[idx]

class MyIterableDataset(IterableDataset):
    def __init__(self, texts, tokenizer, stack = True):
        self.examples = []
        self.token_count = 0
        for text in tqdm(texts, total=len(texts)):
            ids = tokenizer.encode(text, add_special_tokens=False)
            self.examples.append(torch.tensor(ids).long())
            self.token_count += len(ids)
        # if stack:
        #     self.examples = torch.cat(self.examples, 0)

    def __iter__(self):
        for item in self.examples:
            yield {"tokens": item} 
    def __len__(self):
        return len(self.examples)

def load_defined_model(configs, load_path = None):
    cfg = HookedTransformerConfig(
    n_layers = configs['n_layers'],
    d_model = configs['d_model'],
    d_head = configs['d_model']//configs['n_head'],
    n_ctx = configs['max_length'],
    d_vocab = configs['vocab_size'], # 50257,  
    seed = 42,
    act_fn="gelu", 
    # use_wandb=False,
    # wandb_project=None,
    init_mode = 'gpt2',
    positional_embedding_type = configs['positions']
)
    print(configs['d_model'])   
    model = HookedTransformer(cfg) # Assuming you want to load the model on GPU 0
    if load_path is not None:
        state_dict = torch.load(load_path, map_location='cpu')  
        model.load_state_dict(state_dict)
    return model