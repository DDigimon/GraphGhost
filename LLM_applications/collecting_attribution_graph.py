import config_env
import os
import gc

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import sys
import json

import random
import numpy as np
import torch
import re
import pickle
from model_load import load_model, model_name_func
import networkx as nx
sys.path.append('../')
from circuit_tracer.replacement_model import ReplacementModel
from circuit_tracer.configs import Configs
from circuit_tracer.graph import Graph, prune_graph
from circuit_tracer import attribute
from circuit_tracer.utils.create_graph_files import create_nodes, create_used_nodes_and_edges

def translate_node_ids(graph, node_id, given_type):
    node_id = node_id.split('_')
    if given_type == 'embedding':
        vocab_id = int(node_id[1])
        pos = int(node_id[2])
        id = tokenizer.decode(vocab_id)
        layer = 'Emb'
    if given_type == "mlp reconstruction error":
        _, layer, pos = int(node_id[0]), int(node_id[1]), int(node_id[2])
        id = 'block_inner'
    if given_type == "logit":
        layer, vocab_id, pos = int(node_id[0]), int(node_id[1]), int(node_id[2])
        id = tokenizer.decode(vocab_id)
    if given_type == "cross layer transcoder":
        layer, feat_idx, pos = int(node_id[0]), int(node_id[1]), int(node_id[2])
        id = tokenizer.decode(graph.input_tokens[pos])
    return id, pos, layer

def nodes_at_n_hop(G, source, n, filter = ''):
    current_layer = set([source])
    all_traversed = set([source])
    for _ in range(n):
        next_layer = set()
        for node in current_layer:
            # print(f"Expanding node: {node}")
            if node not in G:
                continue
            next_layer.update(G.successors(node))
            all_traversed.update(G.successors(node))
        # print(f"Current layer: {current_layer}, Next layer: {next_layer}")
        current_layer = next_layer
    if filter != '':
        all_traversed = {node for node in all_traversed if node.split('_')[1] == filter}
    return all_traversed

def implicit_route(prompt_str, ans_pos):
    graph = attribute(
            prompt=prompt_str,
            model=model,
            # input_ids=input_ids,
            input_ids=None,
            max_n_logits=max_n_logits,
            desired_logit_prob=desired_logit_prob,
            batch_size=batch_size,
            max_feature_nodes=max_feature_nodes,
            offload=offload,
            verbose=verbose,
            print_log=False,
        )

        
    node_threshold = node_ratio
    edge_threshold = edge_ratio# 0.99


    node_mask, edge_mask, cumulative_scores = (
        el.cpu() for el in prune_graph(graph, node_threshold, edge_threshold)
    )
    scan = graph.scan
    nodes = create_nodes(graph, node_mask, tokenizer, cumulative_scores, scan)
    used_nodes, used_edges = create_used_nodes_and_edges(graph, nodes, edge_mask)
    nodes_dicts = {}
    for n in used_nodes:
        nodes_dicts[n.node_id] = {}
        nodes_dicts[n.node_id]['features'] = n.feature_type
        nodes_dicts[n.node_id]['clerp'] = n.clerp
    
    logit_nodes = []
    result_graph = {}
    for e in used_edges:
        if nodes_dicts[e['source']]['features'] == "mlp reconstruction error": continue
        if nodes_dicts[e['target']]['features'] == "mlp reconstruction error": continue
        if nodes_dicts[e['target']]['features'] == "logit" and nodes_dicts[e['source']]['features'] == 'embedding': continue
        # if nodes_dicts[e['source']]['features'] == "cross layer transcoder":
        weights = e['weight']
        if weights<0. :continue
        source_node, source_pos, source_layer = translate_node_ids(graph, e['source'], nodes_dicts[e['source']]['features'])
        target_node, target_pos, target_layer = translate_node_ids(graph, e['target'], nodes_dicts[e['target']]['features'])
        # print(f"Edge from {source_node} {e['source']} ({nodes_dicts[e['source']]['features']}, pos: {source_pos}, layer: {source_layer}) to {target_node} {e['target']} ({nodes_dicts[e['target']]['features']}, pos: {target_pos}, layer: {target_layer}) with weight {weights}")
        
        start_node = f'{source_node}_{source_layer}_{source_pos}'
        end_node = f'{target_node}_{target_layer}_{target_pos}'
        if end_node not in result_graph:
            result_graph[end_node] = {}
        if start_node not in result_graph[end_node]:
            result_graph[end_node][start_node] = 0
        result_graph[end_node][start_node] += weights
        
        if nodes_dicts[e['target']]['features'] == 'logit':
            logit_nodes.append(end_node)
    
    result_graph = {outer_k: {inner_k: inner_v 
            for inner_k, inner_v in outer_v.items() if inner_v > 0} 
    for outer_k, outer_v in result_graph.items()}
    result_graph = {k: v for k, v in result_graph.items() if v}
    result_graph_G = nx.DiGraph()
    
    for key in result_graph.keys():
        for inner_key in result_graph[key].keys():
            result_graph_G.add_edge(key, inner_key, weight=result_graph[key][inner_key])
    
    marked_pos = []
    repeat_check = set()
    for target_node in list(set(logit_nodes)):    
        target_layers = nodes_at_n_hop(result_graph_G, target_node, n_layer + 2)
        for t in target_layers:
            # TODO some text may include '_' may change a method to get features in the future version
            if len(t.split('_')) != 3: continue
            word, layer, pos = t.split('_')
            if int(pos) > ans_pos: 
                tmp_t = f'{word}_{layer}_{pos}_1'
            else:tmp_t = f'{word}_{layer}_{pos}_0'
            if tmp_t in repeat_check:continue
            repeat_check.add(tmp_t)
            if int(pos) > ans_pos and int(pos) not in marked_pos:
                marked_pos.append(int(pos))
    
    del graph
    gc.collect() 
    torch.cuda.empty_cache()
    # print(marked_pos)
    return marked_pos, repeat_check, result_graph_G


### set seed
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

### parameter initialize

if __name__ =='__main__':
    edge_ratio = 0.5
    node_ratio = 0.4

    name = 'Qwen3-0.6B'
    cache_root = "huggingface_path"
    
    
    cuda_range = [1]
    cuda_list = [f'cuda:{i}' for i in cuda_range]


    model_name = model_name_func(name)
    model = load_model(name, device=f'cuda:{cuda_range[0]}')

    tokenizer = model.tokenizer
    n_layer = model.cfg.n_layers
    d_model = model.cfg.d_model
    split_layer = int(n_layer/len(cuda_range))

    plan = []
    for s, cuda_id in enumerate(cuda_range):
        end = (s + 1)*split_layer
        if end>n_layer:
            end = n_layer
        plan.append((torch.device(f'cuda:{cuda_id}'),range(s*split_layer,end)))

    device_map = {}
    for dev, layers in plan:
        for L in layers:
            device_map[L] = dev

    ### configs loading
    acts_func = 'relu'
    train_dataset_name = 'maw'
    
    save_path = f'save/{name}_{train_dataset_name}_{edge_ratio}_{node_ratio}/'
    save_graph_path = f'save_graph/graph_{name}_{train_dataset_name}_{edge_ratio}_{node_ratio}/'

    transcoder_save_path = f'transcoder_model/{acts_func}_{name}_{train_dataset_name}'
    transcoder_model_save_path = f'transcoder_model/{acts_func}_{name}_{train_dataset_name}'

    transcoder_config_path = os.path.join(transcoder_save_path,'config.json')
    configs = Configs.init_load(transcoder_config_path)
    configs.set_device_map(device_map)

    del model
    model = ReplacementModel.from_self_pretrained_and_transcoders(cfg=configs,model_name=name, model_path = cache_root, transcoders_path = transcoder_model_save_path)
    model.forward = model.forward_sharded
    model.set_device_map(device_map)



    model.eval()  # Set the model to evaluation mode

    save_json = f'./data_{configs.pattern_name}/{name}_answer.json'
    with open(save_json,'r') as f:
        test_data = json.load(f)



    match_collect = {}
    for idx, data in enumerate(test_data):
        text = str(data['gold_ans'])
        match_collect[idx] = text
                
                
    max_n_logits = 10   # How many logits to attribute from, max. 
    desired_logit_prob = 0.95  # Attribution will attribute from the minimum number of logits needed to reach this probability mass (or max_n_logits, whichever is lower)
    max_feature_nodes = 2048  # Only attribute from this number of feature nodes, max. Lower is faster, but you will lose more of the graph. None means no limit.
    batch_size=128
    offload = 'cpu'# 'disk' if IN_COLAB else 'cpu' # Offload various parts of the model during attribution to save memory. Can be 'disk', 'cpu', or None (keep on GPU)
    verbose = True 

    count = 0

    print(len(test_data))
    
    # ⚠️ Warning: change you own idx range in test set
    start = 000
    end = 100
    test_data = test_data[start:end]

    tokenizer_Answer = tokenizer('Answer')['input_ids'][0]

    for idx, data in enumerate(test_data):
        # files = [int(f.split('.')[0]) for f in os.listdir(save_path) if os.path.isfile(os.path.join(save_path, f))]
        real_id = start + idx

        for node_thre in [node_ratio]:
            overall_acc = []
            layer_counts = {}
            counts = 0
            
            prompt = "Question: " + data['question'] + "\nLet's think step by step\nAnswer:\n"
            ans = data['ans']
            
            try:
                tokens = tokenizer(ans, return_offsets_mapping=True)
                offsets = tokens['offset_mapping']
                ids = tokens['input_ids']
                ans_position = ids.index(tokenizer_Answer)
                position_lookup = {}
                if real_id not in match_collect: continue
                last_pos_ans = ans.rfind(match_collect[real_id])
                ans_token = tokenizer(ans[:last_pos_ans])['input_ids']
                dicts = {}
                print(f"    current_idx:{idx}")
                marked_pos, repeat_check, graph = implicit_route(ans[:last_pos_ans], ans_position)
                dicts['**ans**'] = match_collect[real_id]
                dicts[ans[last_pos_ans:last_pos_ans+1]] = {}
                dicts[ans[last_pos_ans:last_pos_ans+1]]['last'] = repeat_check
                graph_list = [graph]

                i = 0
                while i < len(marked_pos):
                    m = marked_pos[i]
                    words = ans[offsets[m][0]:offsets[m][1]]
                    
                    if words not in dicts:
                        dicts[words] = {}
                    new_marked_pos, new_repeat_check, new_graph = implicit_route(ans[:offsets[m][0]], ans_position)
                    dicts[words][m] = new_repeat_check
                    graph_list.append(new_graph)
                    added = 0
                    for new_m in new_marked_pos:
                        if new_m not in marked_pos:
                            marked_pos.append(new_m)
                            added += 1  
                    
                    i += 1
                    marked_pos_lens = len(marked_pos)
                    print(f"idx:{i}, marked:{marked_pos_lens}")
                    
                print(f"idx:{idx}, marked:{len(marked_pos)}")
                if len(marked_pos) <=0 :continue
                with open(save_path + f'{real_id}.pkl', 'wb') as f:
                    pickle.dump(dicts, f)
                with open(save_graph_path + f'{real_id}.pkl', 'wb') as f:
                    pickle.dump(graph_list, f)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print("Error message:", e)           
        count += 1