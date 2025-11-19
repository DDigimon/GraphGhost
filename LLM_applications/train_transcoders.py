import os
import gc

os.environ["TOKENIZERS_PARALLELISM"] = "false"
cache_root = "/egr/research-dselab/daixinna/shared/huggingface_path"

os.environ["HF_HOME"] = cache_root  


import torch
import os
import sys
from tqdm import tqdm
import json
from torch.utils.data import IterableDataset

from model_load import load_model, model_name_func

sys.path.append('../')
from circuit_tracer.configs import Configs
from circuit_tracer.transcoder.activation_functions import JumpReLU, Relu, TopK
from circuit_tracer.transcoder.single_layer_transcoder import SingleLayerTranscoder
from circuit_tracer.transcoder.activations_store import ActivationsStore
from circuit_tracer.transcoder_training import train_sae_on_language_model

class MyIterableDataset(IterableDataset):
    def __init__(self, texts, tokenizer):
        self.examples = []
        self.token_count = 0
        for text in tqdm(texts, total=len(texts)):
            ids = tokenizer.encode(text, add_special_tokens=False)
            self.examples.append(torch.tensor(ids).long())
            self.token_count += len(ids)

    def __iter__(self):
        for item in self.examples:
            yield {"tokens": item} 
    def __len__(self):
        return len(self.examples)



device = 'cuda:0'


acts_func = 'relu'
from_pretrained = False
name = 'Qwen3-0.6B'
model_name = model_name_func(name)
model = load_model(name, device=device)

n_layer = model.cfg.n_layers
d_model = model.cfg.d_model
    
tokenizer = model.tokenizer
train_dataset_name = 'maw'
l1_co = 0.00005
max_length = 200
total_training_tokens = 5_000_000

transcoder_save_path = f'/egr/research-dselab/shared/transcoder_model/{acts_func}_{name}_{train_dataset_name}'

activate_cache_dir = '/egr/research-dselab/shared/daixinna/huggingface_path/activations_cache'

if os.path.exists(activate_cache_dir) == False:
    os.makedirs(activate_cache_dir)
if os.path.exists(transcoder_save_path) == False:
    os.makedirs(transcoder_save_path)

acts_func = 'relu'
transcoder_config = {}
transcoder_config['model_name'] = model_name
transcoder_config['train_dataset_name'] = train_dataset_name
transcoder_config['dataset_path'] = None
transcoder_config['pattern_name'] = 'maw'
transcoder_config['method'] = 'real'
transcoder_config['is_tokenized'] = True
transcoder_config['max_length'] = max_length
transcoder_config['total_train_num'] = total_training_tokens
transcoder_config['acts_func'] = acts_func

# turn epoch if the training set is small while MSE still high
transcoder_config['epoch'] = 1
transcoder_config['batch_size'] = 1024
transcoder_config['train_batch_size'] = 1024
transcoder_config['resample_batches'] = 1024
# turn this parameters when out of memory (very efficient)
transcoder_config['n_batches_in_buffer'] = 32
transcoder_config['store_batch_size'] = 16
# turn this parameter to balance the MSE and L1 loss
transcoder_config['l1_coefficient'] = l1_co

# decompose the features to sparse space (can turn the parameters)
transcoder_config['d_transcoder'] = d_model * 2

# large dead feature window is efficient when large vocab size
transcoder_config['dead_feature_window'] = d_model
transcoder_config['dead_feature_threshold'] = 1e-8
# do not be too small. Better increasing training num instead of large lr
transcoder_config['lr'] = 1e-3

base_model_configs = {}
base_model_configs['d_model'] = d_model
base_model_configs['max_length'] = max_length



save_json = f'./data_{train_dataset_name}/{name}_answer.json'
with open(save_json,'r') as f:
    test_data = json.load(f)
    
training_corpus = []
for data in test_data:
    training_corpus.append(data['ans'])
valid_corpus = training_corpus[:10]


train_dataset = MyIterableDataset(training_corpus, tokenizer)
valid_dataset = MyIterableDataset(valid_corpus, tokenizer)





for target_layer in range(0, n_layer):
    configs = Configs.init_setup(target_layer, transcoder_config, base_model_configs, transcoder_save_path, device, activate_cache_dir)
    model = model.to(configs.act_store_device)
    
    configs.tokenizer_name = None
    
    activation_stores = ActivationsStore(configs, model, dataset=train_dataset, tokenizer=tokenizer)

    if acts_func == 'relu':
        transcoder = SingleLayerTranscoder(configs, Relu())# JumpReLU(0.0, 0.1))

    record_scores = train_sae_on_language_model(configs, model, transcoder, activation_stores, use_eval=False)
    print(f"Target Layer: {target_layer}")
    # del to save the cuda space
    del activation_stores
    del transcoder
    gc.collect()
    torch.cuda.empty_cache()
    
transcoder_config_path = os.path.join(transcoder_save_path,'config.json')
configs.save(transcoder_config_path)