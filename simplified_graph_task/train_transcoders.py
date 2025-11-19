from transformer_lens import HookedTransformer, HookedTransformerConfig
from dataclasses import dataclass
import torch
import os
import sys
from abc import ABC
import json
import pickle
from transformers import PreTrainedTokenizerFast
from utils import MyIterableDataset

sys.path.append('../')
from circuit_tracer.configs import Configs
from circuit_tracer.transcoder.activation_functions import JumpReLU, Relu, TopK
from circuit_tracer.transcoder.single_layer_transcoder import SingleLayerTranscoder
from circuit_tracer.transcoder.activations_store import ActivationsStore
from circuit_tracer.transcoder_training import train_sae_on_language_model

def load_defined_model(configs, load_path = None):
    cfg = HookedTransformerConfig(
    n_layers = configs['n_layers'],
    d_model = configs['d_model'],
    d_head = configs['d_model']//configs['n_head'],
    n_ctx = configs['max_length'],
    d_vocab = configs['vocab_size'], # 50257,  
    seed = 42,
    act_fn="gelu", 
    init_mode = 'gpt2',
    positional_embedding_type = configs['positions']
    )
    model = HookedTransformer(cfg) # Assuming you want to load the model on GPU 0
    if load_path is not None:
        state_dict = torch.load(load_path, map_location='cpu')  
        model.load_state_dict(state_dict)
    return model


base_path=f'base'
activate_cache_dir = 'activations_cache'
dataset_path = os.path.join(base_path,'datasets')

device = 'cuda:0'
n = 10
pattern_name = f'sync{n}'
method = 'path'
n_layer = 5
density = 0.8
edge_remove_density = 0.4
d_hid = 128 # 1536# 768# 192
n_hid = 12

tokenizer_path = os.path.join(dataset_path, f"{pattern_name}_{method}_baby_tokenizer.json")
base_model_configs_save_path = os.path.join(base_path,f'baby_{pattern_name}_{method}_{n_layer}_{d_hid}_configs.json')
out_dir = os.path.join(base_path, f"models/{pattern_name}_{method}_{n_layer}")
model_path = os.path.join(out_dir, f"{n_layer}_{d_hid}_model_weights.pt")

with open(os.path.join(f'{dataset_path}',f'train_{pattern_name}_{density}_{edge_remove_density}_{method}.pkl'),'rb') as f:
    training_corpus = pickle.load(f)
with open(os.path.join(f'{dataset_path}',f'valid_{pattern_name}_{density}_{edge_remove_density}_{method}.pkl'),'rb') as f:
    valid_corpus = pickle.load(f)

with open(base_model_configs_save_path, "r") as f:
    base_model_configs = json.load(f)
train_dataset_name = f'{pattern_name}_{method}'
tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)


train_dataset = MyIterableDataset(training_corpus, tokenizer,stack=True)
valid_dataset = MyIterableDataset(valid_corpus, tokenizer,stack=True)
total_token = train_dataset.token_count
print('loaded train dataset:', len(train_dataset), 'valid dataset:', len(valid_dataset))

model = load_defined_model(base_model_configs, load_path = model_path)
model = model.to(device)
print('loaded model:', model_path)


total_training_tokens = 1_000_000 * 60
if total_token < total_training_tokens:
    total_training_tokens = total_token
print('total token:', total_token)

acts_func = 'relu'
transcoder_config = {}
transcoder_config['model_name'] = 'gpt2'
transcoder_config['train_dataset_name'] = train_dataset_name
transcoder_config['dataset_path'] = base_path
transcoder_config['pattern_name'] = pattern_name
transcoder_config['method'] = method
transcoder_config['is_tokenized'] = True
transcoder_config['max_length'] = base_model_configs['max_length']
transcoder_config['total_train_num'] = total_training_tokens
transcoder_config['acts_func'] = acts_func

# turn epoch if the training set is small while MSE still high
transcoder_config['epoch'] = 1
transcoder_config['batch_size'] = 4096
transcoder_config['train_batch_size'] = 4096
transcoder_config['resample_batches'] = 1024
# turn this parameters when out of memory (very efficient)
transcoder_config['n_batches_in_buffer'] = 128
transcoder_config['store_batch_size'] = 32
# turn this parameter to balance the MSE and L1 loss
transcoder_config['l1_coefficient'] = 0.001

# decompose the features to sparse space (can turn the parameters)
transcoder_config['d_transcoder'] = base_model_configs['d_model'] * 2

# large dead feature window is efficient when large vocab size
transcoder_config['dead_feature_window'] = 50
transcoder_config['dead_feature_threshold'] = 1e-8
# do not be too small. Better increasing training num instead of large lr
transcoder_config['lr'] = 1e-3

transcoder_save_path = f'transcoder_model/{acts_func}_{pattern_name}_{method}_{n_layer}_{d_hid}'


if os.path.exists(transcoder_save_path) == False:
    os.makedirs(transcoder_save_path)


for target_layer in range( n_layer):
    configs = Configs.init_setup(target_layer, transcoder_config, base_model_configs, transcoder_save_path, device, activate_cache_dir)
    model = model.to(configs.act_store_device)
    configs.tokenizer_name = None
    if configs.from_pretrained:
        activation_stores = ActivationsStore(configs, model, dataset=train_dataset)
    else:
        activation_stores = ActivationsStore(configs, model, dataset=train_dataset,tokenizer=tokenizer)
    # TODO: we only evaluate on relu function. The others are not been explored.
    if acts_func == 'relu':
        transcoder = SingleLayerTranscoder(configs, Relu())
    if acts_func == 'jump':
        transcoder = SingleLayerTranscoder(configs, JumpReLU(0.0, 0.1))
    if acts_func == 'topk':
        transcoder = SingleLayerTranscoder(configs, TopK( k = 4))

    _, record_scores = train_sae_on_language_model(configs, model, transcoder, activation_stores, use_eval=False)
    print(f"Target Layer: {target_layer}, Record Scores: {record_scores}")

configs.save(os.path.join(base_path,'transcoder_configs.json'))
