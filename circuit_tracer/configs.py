from dataclasses import dataclass, field
import torch
import os
import sys
from abc import ABC
from typing import Optional, Dict
import json
import yaml



@dataclass
class Configs(ABC):
    target_layer: int
    out_hook_point_layer: int
    # max_layer: int
    d_in: int
    epoch: int
    d_out: int
    d_transcoder: int  # Dimension of the transcoder output
    context_size: int
    hook_point_head_index: Optional[int]
    
    
    train_batch_size: int
    dataset_path: str
    is_dataset_tokenized: bool
    
    hook_point: str
    out_hook_point: str
    is_transcoder: bool
    is_sae: bool
    use_cached_activations: bool
    cached_activations_path: str
    
    n_batches_in_buffer: int
    total_training_tokens: int
    store_batch_size: int
    
    act_store_device: str
    device: str

    
    seed: int
    # dtype: torch.dtype # = torch.float32
    b_dec_init_method: str
    l1_coefficient: float
    
    model_name: str
    dataset_name: str
    pattern_name: str
    method: str
    max_length: int  # Default value, can be overridden
    activate_func:str
    
    batch_size: int
    lr: float
    dead_feature_window: int
    dead_feature_estimation_method: str
    dead_feature_threshold: float
    resample_batches: int
    is_sparse_connection: bool
    checkpoint_path: str
    lr_scheduler_name: str
    lr_warm_up_steps: int
    # max_layer: int
    from_pretrained: bool
    tokenizer_name: Optional[str] = None
    device_map: Dict[int, str] = field(default_factory=dict)
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.__dict__, f)
            
    def load_json_file(self, path: str):
        with open(path, 'r') as f:
            config_dict = json.load(f)
        for key, value in config_dict.items():
            setattr(self, key, value)
    
    @staticmethod
    def load(path):
        with open(path, 'r') as f:
            config_dict = json.load(f)
            
                
        return Configs(**config_dict)

    ### for initialization of configs
    def init_load(path):
        with open(path, 'r') as f:
            config_dict = json.load(f)
                
        return Configs(**config_dict)
    
    def init_setup(target_layer, transcoder_config, base_model_configs, transcoder_saves, training_device, activate_cache_dir):
        config_dict = {
            "target_layer": target_layer,
            "out_hook_point_layer": target_layer,
            "epoch": transcoder_config['epoch'],
            "d_in": base_model_configs['d_model'],
            "d_out": base_model_configs['d_model'],
            "d_transcoder": transcoder_config['d_transcoder'],
            "train_batch_size": transcoder_config['train_batch_size'],
            "context_size": base_model_configs['max_length'],
            "dataset_path": transcoder_config['dataset_path'],
            "is_dataset_tokenized": transcoder_config['is_tokenized'],
            "pattern_name": transcoder_config['pattern_name'],
            "method": transcoder_config['method'],
            "max_length": transcoder_config['max_length'],

            "hook_point": f"blocks.{target_layer}.ln2.hook_normalized",
            "out_hook_point": f"blocks.{target_layer}.hook_mlp_out",

            "is_transcoder": True,
            "is_sae": False,
            "use_cached_activations": False,
            "cached_activations_path": activate_cache_dir,

            "n_batches_in_buffer": transcoder_config['n_batches_in_buffer'],
            "total_training_tokens": transcoder_config['total_train_num'],
            "store_batch_size": transcoder_config['store_batch_size'],

            "act_store_device": training_device,
            "device": training_device,
            "seed": 42,
            "hook_point_head_index": None,

            "b_dec_init_method": "mean",
            "l1_coefficient": transcoder_config['l1_coefficient'],
            "activate_func": transcoder_config['acts_func'],

            "model_name": transcoder_config['model_name'],
            "dataset_name": "graph",

            "batch_size": transcoder_config['batch_size'],
            "lr": transcoder_config['lr'],
            "dead_feature_window": transcoder_config['dead_feature_window'],
            "dead_feature_estimation_method": "no_fire",
            "dead_feature_threshold": transcoder_config['dead_feature_threshold'],
            "resample_batches": transcoder_config['resample_batches'],
            "is_sparse_connection": False,

            "checkpoint_path":transcoder_saves,

            "lr_scheduler_name": "constantwithwarmup",
            "lr_warm_up_steps": 5000,
            "from_pretrained": False,
            }
        return Configs(**config_dict)

    def set_device_map(self, device_map):
        self.device_map = device_map
    
