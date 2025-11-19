import os
os.environ["WANDB_MODE"] = "disabled"
os.environ['TMPDIR']='/egr/research-dselab/daixinna/temp'
os.environ["HF_HOME"] = '/egr/research-dselab/daixinna/shared/huggingface_path'
from transformer_lens import HookedTransformerConfig
from transformer_lens import HookedTransformer
import torch
import pickle
import os
import math
from tqdm import tqdm
from contextlib import nullcontext
import json
from transformers import PreTrainedTokenizerFast
import random
import torch.nn.functional as F
import inspect
import numpy as np
from utils import MyDataset, build_cfg






def configure_optimizers(model, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in model.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer







def setup_train_data(dataset_path, pattern_name, density, edge_remove_density, method):
    with open(os.path.join(f'{dataset_path}',f'train_{pattern_name}_{density}_{edge_remove_density}_{method}.pkl'),'rb') as f:
        training_corpus = pickle.load(f)
    with open(os.path.join(f'{dataset_path}',f'valid_{pattern_name}_{density}_{edge_remove_density}_{method}.pkl'),'rb') as f:
        valid_corpus = pickle.load(f)
    print(training_corpus[:5])

    tokenizer_path = os.path.join(dataset_path, f"{pattern_name}_{method}_baby_tokenizer.json")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    vocab_dict = tokenizer.get_vocab()
    print('vocab size: ',len(set(vocab_dict)), tokenizer.vocab_size)
    train_dataset = MyDataset(training_corpus, tokenizer,stack=True)
    valid_dataset = MyDataset(valid_corpus, tokenizer,stack=True)
    print(len(train_dataset),len(valid_dataset))
    return train_dataset, valid_dataset, tokenizer, len(set(vocab_dict))

def setup_model(base_path, vocab_size, n_layer, d_hid, n_hid, pattern_name, method, device, max_length):

    configs = {"n_layers": n_layer,
            "d_model": d_hid, # 192, # 384,
            "n_head": n_hid,
            "max_length": max_length,
            "vocab_size": vocab_size,
            "pattern_name": pattern_name,
            "method": method,
            "positions": 'rotary',
            'act_fn': 'gelu',
            'seed': 42,
            'init_mode': 'gpt2'}


    cfg = build_cfg(configs)

    configs_save_path = os.path.join(base_path,f'baby_{pattern_name}_{method}_{n_layer}_{d_hid}_configs.json')
    with open(configs_save_path, "w") as f:
        json.dump(configs, f)

    model = HookedTransformer(cfg)
    model = model.to(device)
    return model, configs






def training(base_path, model, config, train_dataset, valid_dataset, tokenizer, device,block_size, learning_rate=1e-3, batch_size=1024, max_iters = 4000, beta1=0.9, beta2=0.99, eval_interval = 500, eval_iters = 15, log_interval = 500):
    # TODO currently using a rough version, we can change it 
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' 
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    optimizer = configure_optimizers(model, weight_decay=0.1, learning_rate=learning_rate, betas=(beta1, beta2), device_type=device_type)
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    iter_num = 0
    loss_dicts={}
    loss_dicts['train']={}
    loss_dicts['val']={}
    vocab_dict = tokenizer.get_vocab()
    vocab_size = len(set(vocab_dict))
    def get_random_batch(dataset, batch_size):
        indices = random.choices(range(len(dataset) - block_size), k=batch_size)
        batch = [dataset[i : i + block_size] for i in indices]
        return torch.stack(batch).to(device)

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['val']:
            losses = torch.zeros(eval_iters)
            for k in tqdm(range(eval_iters)):
                batch = get_random_batch(valid_dataset, batch_size)
                with ctx:
                    logits = model(batch)
                    loss = count_loss(logits, batch)
                            
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out
    
    def count_loss(logits, batch):
        loss = F.cross_entropy(
                        logits[:, :-1, :].reshape(-1, vocab_size),
                        batch[:, 1:].reshape(-1),
                        ignore_index=tokenizer.pad_token_id
                    )
        return loss
    pattern_name = config['pattern_name']
    method = config['method']
    n_layer = config['n_layers']
    d_hid = config['d_model']
    out_dir = os.path.join(base_path, f"models/{pattern_name}_{method}_{n_layer}")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    gradient_accumulation_steps = 1

    best_val_loss = float('inf')

    grad_clip = 1.0 
    while True:
        if iter_num % eval_interval == 0:
            losses = estimate_loss()
            loss_dicts['val'][iter_num]=float(losses['val'])
            print(f"step {iter_num}, val loss {losses['val']}")
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                print(f"saving checkpoint to {out_dir}")
                # torch.save(model.state_dict(), os.path.join(out_dir, f"{n_layer}_{d_hid}_model_weights.pt"))

            
        batch = get_random_batch(train_dataset, batch_size)
        logits = model(batch)
        # Shift for next-token prediction
        loss = count_loss(logits, batch)


        scaler.scale(loss).backward()
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        iter_num += 1

        if iter_num % log_interval == 0:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * gradient_accumulation_steps
            loss_dicts['train'][iter_num]=float(lossf)
            print(f"iter {iter_num}: loss {lossf:.4f}")

            with open(os.path.join(out_dir,'val_loss.json'),'w') as f:
                json.dump(loss_dicts,f)
        if iter_num > max_iters:
            break

if __name__ == '__main__':
    device = 'cuda:0'

    n = 10
    pattern_name = f'sync{n}'
    method = 'path'

    n_layer = 5
    density = 0.8
    edge_remove_density = 0.4
    d_hid = 128 # 1536# 768# 192
    n_hid = 12
    block_size = 128
    base_path=f'/egr/research-dselab/shared/daixinna/graph_reasoning/baby_models/'

    dataset_path = os.path.join(base_path,'datasets')

    eval_interval = 250
    log_interval = 250
    
    train_dataset, valid_dataset, tokenizer, vocab_size = setup_train_data(dataset_path, pattern_name, density, edge_remove_density, method)
    model, config = setup_model(base_path, vocab_size, n_layer, d_hid, n_hid, pattern_name, method, device, max_length=block_size)    
    training(base_path, model, config, train_dataset, valid_dataset, tokenizer, device,block_size, max_iters=6000, eval_interval=eval_interval, log_interval=log_interval)    