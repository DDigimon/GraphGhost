from circuit_tracer.transcoder.circuit_analysis import *
from circuit_tracer.transcoder.feature_dashboards import *
from circuit_tracer.transcoder.replacement_ctx import *

from transformer_lens import HookedTransformer, utils
import os
import torch

import einops

from datasets import load_dataset
from huggingface_hub import HfApi
import numpy as np
import pickle
from transformers import PreTrainedTokenizerFast

# from utils import MyDataset, MyIterableDataset

# def load_self_train_dataset(base_path, pattern_name, method, valid_corpus=None):
#     dataset_path = os.path.join(base_path,'datasets')
#     if '_h' in pattern_name:
#         pattern_name = pattern_name.split('_h')[0]
#     if '-' in pattern_name:
#         tokenizer_path = os.path.join(dataset_path, f"{pattern_name.split('-')[0]}_{method}_baby_tokenizer.json")
#     else:
#         tokenizer_path = os.path.join(dataset_path, f"{pattern_name}_{method}_baby_tokenizer.json")
#     tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
#     with open(os.path.join(f'{dataset_path}',f'train_{pattern_name}_{method}.pkl'),'rb') as f:
#         training_corpus = pickle.load(f)
#     with open(os.path.join(f'{dataset_path}',f'valid_{pattern_name}_{method}.pkl'),'rb') as f:
#         valid_corpus = pickle.load(f)
    
#     if 'pad' in base_path:
#         slide_idx = training_corpus[0].split(' ').index('START_A')
#         train_dataset = MyIterableDataset(training_corpus, tokenizer,stack=False)
#         valid_dataset = MyDataset(valid_corpus, tokenizer,stack=False)
#     else:
#         slide_idx = None
#         train_dataset = MyIterableDataset(training_corpus, tokenizer,stack=True)
#         valid_dataset = MyDataset(training_corpus, tokenizer,stack=True)
        
#     return train_dataset, valid_dataset, tokenizer, slide_idx

# def load_self_valid_data(config):
#     base_path = config.dataset_path
#     pattern_name = config.pattern_name
#     method = config.method
#     dataset_path = os.path.join(base_path,'datasets')
#     if '_h' in pattern_name:
#         pattern_name = pattern_name.split('_h')[0]
#     if '-' in pattern_name:
#         tokenizer_path = os.path.join(dataset_path, f"{pattern_name.split('-')[0]}_{method}_baby_tokenizer.json")
#     else:
#         tokenizer_path = os.path.join(dataset_path, f"{pattern_name}_{method}_baby_tokenizer.json")
#     tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
#     with open(os.path.join(f'{dataset_path}',f'valid_{pattern_name}_{method}.pkl'),'rb') as f:
#         valid_corpus = pickle.load(f)
    
#     valid_dataset = MyDataset(valid_corpus, tokenizer,stack=True)
#     return valid_dataset
    

def tokenize_and_concatenate(
    dataset,
    tokenizer,
    streaming = False,
    max_length = 1024,
    column_name = "text",
    add_bos_token = True,
):
    for key in dataset.features:
        if key != column_name:
            dataset = dataset.remove_columns(key)

    if tokenizer.pad_token is None:
        # We add a padding token, purely to implement the tokenizer. This will be removed before inputting tokens to the model, so we do not need to increment d_vocab in the model.
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    # Define the length to chop things up into - leaving space for a bos_token if required
    if add_bos_token:
        seq_len = max_length - 1
    else:
        seq_len = max_length

    def tokenize_function(examples):
        text = examples[column_name]
        # Concatenate it all into an enormous string, separated by eos_tokens
        full_text = tokenizer.eos_token.join(text)
        # Divide into 20 chunks of ~ equal length
        num_chunks = 20
        chunk_length = (len(full_text) - 1) // num_chunks + 1
        chunks = [
            full_text[i * chunk_length : (i + 1) * chunk_length]
            for i in range(num_chunks)
        ]
        # Tokenize the chunks in parallel. Uses NumPy because HuggingFace map doesn't want tensors returned
        tokens = tokenizer(chunks, return_tensors="np", padding=True)[
            "input_ids"
        ].flatten()
        # Drop padding tokens
        tokens = tokens[tokens != tokenizer.pad_token_id]
        num_tokens = len(tokens)
        num_batches = num_tokens // (seq_len)
        # Drop the final tokens if not enough to make a full sequence
        tokens = tokens[: seq_len * num_batches]
        tokens = einops.rearrange(
            tokens, "(batch seq) -> batch seq", batch=num_batches, seq=seq_len
        )
        if add_bos_token:
            prefix = np.full((num_batches, 1), tokenizer.bos_token_id)
            tokens = np.concatenate([prefix, tokens], axis=1)
        return {"tokens": tokens}

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=[column_name],
    )
    #tokenized_dataset.set_format(type="torch", columns=["tokens"])
    return tokenized_dataset

def _load_data(model, configs):
    dataset = load_dataset(configs.dataset_path , split='train', streaming=True)
    dataset = dataset.shuffle(seed=42, buffer_size=10_000)
    tokenized_owt = tokenize_and_concatenate(dataset, model.tokenizer, max_length=128, streaming=True)
    tokenized_owt = tokenized_owt.shuffle(42)
    tokenized_owt = tokenized_owt.take(12800*2)
    owt_tokens = np.stack([x['tokens'] for x in tokenized_owt])
    owt_tokens_torch = torch.from_numpy(owt_tokens).cuda()
    return owt_tokens_torch

    
def eval_transcoder_l0_ce(configs, model,  transcoder, num_batches=100, batch_size=128):
    l0s = []
    transcoder_losses = []
    if batch_size > configs.max_length:
        batch_size = configs.max_length
    if configs.from_pretrained:
        all_tokens = _load_data(model, configs)
    else:
        all_tokens = load_self_valid_data(configs)
    
    with torch.no_grad():
        for batch in tqdm.tqdm(range(0, num_batches)):
            torch.cuda.empty_cache()
            # if configs.from_pretrained:
            # print('sizes', batch, batch_size)
            cur_batch_tokens = all_tokens[batch*batch_size:(batch+1)*batch_size]
            # print(f"Batch {batch} tokens shape: {cur_batch_tokens.shape}")
            # else: 
            #     cur_batch_tokens = 
            #     cur_batch_tokens = cur_batch_tokens.to(model.cfg.device)
            with TranscoderReplacementContext(configs, model, [transcoder]):
                cur_losses, cache = model.run_with_cache(cur_batch_tokens, return_type="loss", names_filter=[transcoder.cfg.hook_point])
                # measure losses
                transcoder_losses.append(utils.to_numpy(cur_losses))
                # measure l0s
                acts = cache[configs.hook_point]
                binarized_transcoder_acts = 1.0*(transcoder(acts)[1] > 0)
                l0s.append(
                    (binarized_transcoder_acts.reshape(-1, binarized_transcoder_acts.shape[-1])).sum(dim=1).mean().item()
                )

    return {
        'l0s': np.mean(l0s),
        'ce_loss': np.mean(transcoder_losses)
    }