import os
os.environ['TMPDIR']='tmp'

import torch
import pickle
import os
import json
import random
import torch
from transformers import PreTrainedTokenizerFast
# from redefined_datasets import MyTestDataset
import pickle
from tqdm import tqdm

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import to_networkx
from utils import MyTestDataset, load_defined_model




def loading(base_path, pattern_name, method, density, edge_remove_density, d_hid, n_layer,device):
    dataset_path = os.path.join(base_path,'datasets')
    tokenizer_path = os.path.join(dataset_path, f"{pattern_name}_{method}_baby_tokenizer.json")
    with open(os.path.join(f'{dataset_path}',f'test_{pattern_name}_{density}_{edge_remove_density}_{method}.pkl'),'rb') as f:
        test_corpus = pickle.load(f)


    with open(os.path.join(base_path, f'data/{pattern_name}-{density}.pkl'),'rb') as f:
        G = pickle.load(f)

    configs_save_path = os.path.join(base_path,f'baby_{pattern_name}_{method}_{n_layer}_{d_hid}_configs.json')
    with open(configs_save_path, "r") as f:
        configs = json.load(f)
    print(configs)
    
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    model_path = os.path.join(base_path, f"models/{pattern_name}_{method}_{n_layer}", f"{n_layer}_{d_hid}_model_weights.pt")
    model = load_defined_model(configs, load_path = model_path)

    model = model.to(device)
    model.eval()

    key_word = 'PATH'
    test_data_set = MyTestDataset(test_corpus, tokenizer,key_word)
    return test_data_set, model, tokenizer, G


def check_hops(text, G):
    g_str = text.split("<START>")[1].split("S")[0].strip()
    
    s = text.split('S ')[1].split()[0]
    e = text.split('E ')[1].split()[0]
    a_str = text.split("PATH")[1].split("END_P")[0].split()# .rstrip(',')
    
    flag = True
    
    sample_flag = True
    
    for i in range(len(a_str)-1):
        ans_pairs = [int(a_str[i]), int(a_str[i + 1])]
        if str(ans_pairs[0])+' '+str(ans_pairs[1]) not in g_str:
            sample_flag = False
        if G.has_edge(ans_pairs[0], ans_pairs[1]) == False:
            flag = False
            sample_flag = False
    if a_str[0] != s or a_str[-1] != e:
        flag = False
        sample_flag = False
    return flag, sample_flag
        
def eval(test_data_set, tokenizer, model,device, G):
    acc_list = []
    sample_acc_list = []

    for idx, test_data in tqdm(enumerate(test_data_set), total=len(test_data_set)):
        # print(test_data)

        max_new_tokens = 10
        generated = test_data[0].to(device)  # shape: (1, current_seq_len)
        generated = generated.unsqueeze(0)

        for _ in range(max_new_tokens):
            try:
                logits = model(generated)  # shape: (1, seq_len, d_vocab)
                next_token_logits = logits[0, -1, :]     
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0).unsqueeze(0)  # shape: (1,1)
                # print(generated.shape,next_token_id.shape)
                generated = torch.cat([generated, next_token_id], dim=1)
            except:
                continue
            
        output_text = tokenizer.decode(generated[0], skip_special_tokens=False)
        if idx == 0:
            print(output_text)
        try:
            flags, sample_flag = check_hops(output_text, G)
            acc_list.append(flags)
            sample_acc_list.append(sample_flag)
        except:
            acc_list.append(0)
            sample_acc_list.append(0)
            continue

    print('acc:', sum(acc_list)/len(acc_list))
    # print(sum(sample_acc_list)/len(sample_acc_list))

if __name__ == "__main__":

    device = 'cuda:0'
    base_path=f'base'
    n_layer = 5
    n = 10
    pattern_name = f'sync{n}'
    method = 'path'
    density = 0.8
    edge_remove_density = 0.4
    d_hid = 128
    n_hid = 12
    block_size = 128

    test_data_set, model, tokenizer, G = loading(base_path, pattern_name, method, density, edge_remove_density, d_hid, n_layer,device)
    eval(test_data_set, tokenizer, model,device, G)