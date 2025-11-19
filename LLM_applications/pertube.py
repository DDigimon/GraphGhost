import config_env
import os
import torch
import time

import random
from tqdm import tqdm
import pickle as pkl
import re
import json
import networkx as nx
from model_load import load_model, model_name_func

from datasets import load_dataset






def answer_judge_string(data):
    text = data['gold_ans']
    match = re.search(r"####\s*(\d+)", text)
    if match is not None:
        match = match.group(1)
        if match not in data['ans']:
            return 0, match
    if data['judge'] is not None:
        judge_prompt = data['judge'].lower()
    else:
        return 0, match
    if 'yes' in judge_prompt and 'no' not in judge_prompt:
        return 1, match
    else:
        return 0, match

def extract_layer(node_name):
    try:
        suffix = node_name.split('_')[-1]
        return -1 if suffix.lower().startswith('emb') else int(suffix)
    except:
        return None  

def extract_token(node_name):
    try:
        suffix = node_name.split('_')[0]
        return suffix
    except:
        return None 
    
def get_token_level_graph(data_graph_list, max_depth = 30):
    '''
    several version of building graph. However, the best way to define the graph also need to be explored
    '''
    global_graph = nx.DiGraph()
    global_graph_without_logit = nx.DiGraph()
    global_graph_with_logit = nx.DiGraph()
    inner_graphs = nx.DiGraph()
    counts = 0
    subgraphs = []
    for idx, graph_data in tqdm(enumerate(data_graph_list), total=len(data_graph_list)):
        flag = 1
        if counts == 100:break # select 100 samples to generate the 
        if len(graph_data) <= 5: continue
        
        for g in graph_data:
            subgraph = nx.DiGraph()
            for edge in list(g.edges()):
                try:
                    s_word, s_layer, _ = edge[0].split('_')
                    e_word, e_layer, _ = edge[1].split('_')
                except:
                    continue

                ls = int(s_layer) + 2 if s_layer.isdigit() else 1
                le = int(e_layer) + 2 if e_layer.isdigit() else 1
                d = (max_depth/ls + max_depth/le) / 2
                weights = 1 / (1 + d)

                if ls < max_depth + 1 and le < max_depth+1: 
                    u = f'{e_word}_{e_layer}'
                    v = f'{s_word}_{s_layer}'
                    if global_graph_without_logit.has_edge(u, v):
                        global_graph_without_logit[u][v]['weight'] += 1
                    else:
                        global_graph_without_logit.add_edge(u, v, weight=1)

                if ls == max_depth + 1 or le == max_depth+1: 
                    v = f'{e_word}_{e_layer}'
                    u = f'{s_word}_{s_layer}'
                else:
                    u = f'{e_word}_{e_layer}'
                    v = f'{s_word}_{s_layer}'
                if global_graph.has_edge(u, v):
                    global_graph[u][v]['weight'] += 1
                else:
                    global_graph.add_edge(u, v, weight=1)
                u = f'{e_word}_{e_layer}'
                v = f'{s_word}_{s_layer}'
                if global_graph_with_logit.has_edge(u, v):
                    global_graph_with_logit[u][v]['weight'] += 1
                else:
                    global_graph_with_logit.add_edge(u, v, weight=1)
        counts += 1
    return global_graph, global_graph_without_logit, global_graph_with_logit

def rewire_token_terminal(G,token, max_layer, top_k = 1, top_down = None):

    top_node = token 
    rewired_edge_dict = {}
    for edges in G.edges():
        u, v = edges
        if v == top_node:print(edges)
        if v != top_node:continue
        if 'Emb' in u or 'Emb' in v: continue
        layer_u = extract_layer(u)
        layer_v = extract_layer(v)
        token_u = extract_token(u)
        token_v = extract_token(v)

        if layer_u == max_layer + 1 or layer_v == max_layer + 1: continue
        if layer_u == -1 or layer_v == -1: continue
        new_v = f'{token_v}_{int(layer_v)-1}' if layer_v > 1 else f'{token_v}_{int(layer_v)+1}'
        if u not in rewired_edge_dict:
            rewired_edge_dict[u] = []
        rewired_edge_dict[u].append([v, new_v])
    if len(rewired_edge_dict) == 0:
        rewired_edge_dict[top_node] = [[top_node, top_node]]
    return rewired_edge_dict

def select_random_node(token, limited_layer = -1, layer_control = None, selected_token = None):
    if selected_token is not None:
        ids = selected_token
        layer = layer_control - 1
        random_token = (layer, ids)
        return random_token

    else:
        while True:
            ids = random.choice(token)
            layers = [i for i in range(model.cfg.n_layers)]
            layer = random.choice(layers)
            if layer < limited_layer or layer < 2: continue
            else: limited_layer = -1
            random_token = (layer, ids)
            # print(random_token, limited_layer)
            if random_token not in special_token_collect and limited_layer == -1:
                return random_token
                break

def random_mask(token, dicts):
    random_dicts = {}
    for key in dicts.keys():
        random_token = select_random_node(token)
        limited_layer = random_token[0]
        random_dicts[random_token] = []
        for edges in dicts[key]:
            start = select_random_node(token, limited_layer=limited_layer)
            layer = start[0]
            selected_token = start[1]
            end = select_random_node(token, limited_layer=limited_layer, layer_control=layer, selected_token=selected_token)
            random_dicts[random_token].append((start, end))
    return random_dicts




def generate_gsm8k(data):
    question, ans = data['question'], data['answer']
    prompt = "Question: " + str(question) + "\nLet's think step by step\nAnswer:\n"
    max_token = 300
    return question, prompt, ans, max_token


def expand_dict_with_positions(rewire_dict_char, toks, mode="rewire"):
    dict_pos = {}
    batch_size, seq_len = toks.shape
    
    for (src_layer, src_token), edges in rewire_dict_char.items():
        src_positions = (toks == src_token).nonzero(as_tuple=False)
        for b, pos in src_positions.tolist():
            src_key = (src_layer, pos)
            dict_pos[src_key] = []
            if len(dict_pos) > 50:continue
            if "rewire" in mode:
                for (tgt_layer_old, tgt_token_old), (tgt_layer_new, tgt_token_new) in edges:
                    if tgt_layer_old == "logits":
                        tgt_positions_old = [(b, seq_len - 1)]
                    else:
                        tgt_positions_old = (toks == tgt_token_old).nonzero(as_tuple=False).tolist()

                    if tgt_layer_new == "logits":
                        tgt_positions_new = [(b, seq_len - 1)]
                    else:
                        tgt_positions_new = (toks == tgt_token_new).nonzero(as_tuple=False).tolist()

                    for _, pos_old in tgt_positions_old:
                        for _, pos_new in tgt_positions_new:
                            if len(dict_pos[src_key]) > 50:continue
                            dict_pos[src_key].append(((tgt_layer_old, pos_old),
                                                    (tgt_layer_new, pos_new)))
            elif "delete" in mode:
                for (tgt_layer_old, tgt_token_old), (tgt_layer_new, tgt_token_new) in edges:
                    for idx,t in enumerate(toks[0]):
                        # print(t, tgt_token_old)
                        if t == tgt_token_old:
                            dict_pos[src_key].append((tgt_layer_old, idx))
                    return dict_pos
                    
    return dict_pos


def intervene_with_dict(model, toks, dict_pos, mode="rewire", alpha=1.0):
        """
        mode="rewire": dict_pos[(src_layer, src_pos)] = [((tgt_layer_old, tgt_pos_old), (tgt_layer_new, tgt_pos_new))]
        mode="delete": dict_pos[(src_layer, src_pos)] = [(tgt_layer_old, tgt_pos_old)]
        """
        cache = {}
        hooks = []

        # resid
        def make_save_src(src_layer, src_pos):
            def save_src(x, hook):
                cache[(src_layer, src_pos)] = x[:, src_pos].detach().clone()
                return x
            return save_src

        # rewiring
        def make_replace_tgt(src_layer, src_pos, tgt_layer, tgt_pos):
            def replace_tgt(x, hook):
                if (src_layer, src_pos) not in cache:
                    return x
                src_val = cache[(src_layer, src_pos)]
                # print(x.shape, tgt_pos)
                x = x.clone()
                if alpha == 1.0:
                    x[:, tgt_pos] = src_val
                else:
                    x[:, tgt_pos] = (1 - alpha) * x[:, tgt_pos] + alpha * src_val
                return x
            return replace_tgt

        # delete
        def make_delete_tgt(tgt_layer, tgt_pos):
            def delete_tgt(x, hook):
                x = x.clone()
                x[:, tgt_pos] = 0.0  
                return x
            return delete_tgt

        for src_key, edges in dict_pos.items():
            src_layer, src_pos = src_key
            if  "rewire" in mode:
                hooks.append((f"blocks.{src_layer}.hook_resid_post", make_save_src(src_layer, src_pos)))
                for (_, _), (tgt_layer_new, tgt_pos_new) in edges:
                    if tgt_layer_new == "logits":
                        hooks.append(("ln_final.hook_normalized",
                                    make_replace_tgt(src_layer, src_pos, tgt_layer_new, tgt_pos_new)))
                    else:
                        hooks.append((f"blocks.{tgt_layer_new}.hook_resid_pre",
                                    make_replace_tgt(src_layer, src_pos, tgt_layer_new, tgt_pos_new)))
            elif "delete" in mode:
                for (tgt_layer_old, tgt_pos_old) in edges:
                    # print('deleting', tgt_layer_old, tgt_pos_old)
                    if tgt_layer_old == "logits":
                        hooks.append(("ln_final.hook_normalized", make_delete_tgt(tgt_layer_old, tgt_pos_old)))
                    else:
                        hooks.append((f"blocks.{tgt_layer_old}.hook_resid_pre", make_delete_tgt(tgt_layer_old, tgt_pos_old)))

        with model.hooks(fwd_hooks=hooks):
            out = model(toks)

        return out

if __name__ == '__main__':
    device = '1'
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    name = 'Qwen3-0.6B' 
    train_dataset_name = 'gsm8k'
    top_k = 1
    top_down = None
    selected_method = 'delete'
    token = ' the_17' 

    edge_ratio = 0.5
    node_ratio = 0.4
    alpha = 1
    l1_co = 0.00005
    save_path = f'/egr/research-dselab/shared/daixinna/GLLM/{name}_{train_dataset_name}_{edge_ratio}_{node_ratio}_{l1_co}/'
    save_graph_path = f'/egr/research-dselab/shared/daixinna/GLLM/graph_{name}_{train_dataset_name}_{edge_ratio}_{node_ratio}_{l1_co}/'
    import os
    idx_list = []
    files = [int(f.split('.')[0]) for f in os.listdir(save_path) if os.path.isfile(os.path.join(save_path, f))]
    data_graph_idx = []
    data_graph_list = []
    for files in os.listdir(save_graph_path):
        with open(save_graph_path + f'{files}', 'rb') as f:
            data = pkl.load(f)
        data_graph_list.append(data)
    print(len(files))


    save_json = f'./data_{train_dataset_name}/{name}_answer.json'
    with open(save_json,'r') as f:
        test_data = json.load(f)

    data_idx = []
    data_list = []
    for files in os.listdir(save_path):
        with open(save_path + f'{files}', 'rb') as f:
            data = pkl.load(f)
        data_list.append(data)
        data_idx.append(int(files.split('.')[0]))
    len(data_list)
    model_name =  model_name_func(name)
    model = load_model(name)
    tokenizer = model.tokenizer

    unk_id = tokenizer(' ', return_tensors="pt")["input_ids"][0]
    print('unk id',unk_id)
    saved_file = f'./edge_pertur_data_{train_dataset_name}/'
    if os.path.exists(saved_file) == False:
        os.makedirs(saved_file)

    text_list = []
    count = 0

    G, G_wo_logit, _ = get_token_level_graph(data_graph_list, max_depth=model.cfg.n_layers+2)
    read_rewired_dict = rewire_token_terminal(G,  token, model.cfg.n_layers)
    print(read_rewired_dict)
    print('get rewired dicts')

    # sparsity_list = []
    token_layer_dicts = {}
    for idx, key in enumerate(read_rewired_dict.keys()):
        word, layer = key.split('_')
        tokens = tokenizer(word)['input_ids']
        layer = int(layer)
        token_layer_dicts[(layer, tokens[-1])] = []
        for edge_pairs in read_rewired_dict[key]:
            # print(edge_pairs)
            word_old, layer_old = edge_pairs[0].split('_')
            word_new, layer_new = edge_pairs[1].split('_')
            token_old = tokenizer(word_old)['input_ids']
            token_new = tokenizer(word_new)['input_ids']
            
            layer_old = int(layer_old)
            if layer_old > model.cfg.n_layers: 
                layer_old = model.cfg.n_layers # 'logits'
            layer_new = int(layer_new)
            if layer_new > model.cfg.n_layers: 
                layer_new = model.cfg.n_layers # 'logits'
            token_layer_dicts[(layer, tokens[-1])].append([(layer_old, token_old[-1]), (layer_new, token_new[-1])])
            # break

    print(token_layer_dicts)


    saved_graphs = G
    special_token_collect = []
    for n in list(saved_graphs.nodes):
        if n.split('_')[1].isdigit() == False:continue
        special_token_collect.append(( int(n.split('_')[1]), tokenizer(n.split('_')[0])['input_ids'][-1]))


    cache_root = "/egr/research-dselab/daixinna/shared/huggingface_path"

    dataset = load_dataset("gsm8k", "main", cache_dir=cache_root)
    test = dataset["test"]


    for idx,data in enumerate(test):
        print(idx)
        count += 1
        question, prompt, ans, max_token = generate_gsm8k(data)
        ori_ans = test_data[idx]['ans']
        print('ori_ans',ori_ans)
        question_tokens = tokenizer(prompt, return_tensors="pt")
        input_ids = question_tokens ["input_ids"].clone()
        output_ids = input_ids.clone().cuda()
        for step in tqdm(range(max_token)):
            rewired_dict = expand_dict_with_positions(token_layer_dicts, output_ids, mode=selected_method)
            logits = intervene_with_dict(model, output_ids, rewired_dict, mode=selected_method, alpha=alpha)
            logits = logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)  # greedy decoding
            output_ids = torch.cat([output_ids, next_token], dim=-1)

            if next_token.item() in model.tokenizer.all_special_ids:
                break
        rewired_ans = tokenizer.decode(output_ids[0])
        random_rewired_ans = ''
        print(rewired_ans)
        text_list.append({'question': question, 'gold_ans': test_data[idx]['gold_ans'], 'ans':rewired_ans, 'ori_ans':ori_ans,'random_ans':random_rewired_ans})

