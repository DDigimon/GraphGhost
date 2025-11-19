import networkx as nx
import random
import pickle
import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from transformers import PreTrainedTokenizerFast



def generate_fully_connected_digraph(n):
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if i != j:
                G.add_edge(i, j)
    return G

def subgraph_by_density(G, target_density):
    assert 0 <= target_density <= 1, "Density must be between 0 and 1."
    n = G.number_of_nodes()
    max_edges = n * (n - 1)
    target_num_edges = int(target_density * max_edges)

    all_edges = list(G.edges())
    random.shuffle(all_edges)
    selected_edges = all_edges[:target_num_edges]

    H = nx.DiGraph()
    H.add_nodes_from(G.nodes())
    H.add_edges_from(selected_edges)
    return H

def sample_2hop_subgraph(G, center_node, max_nodes=5, edge_keep_ratio=0.6, edge_remove = False):
    hop1_neighbors = set(G.successors(center_node)).union(G.predecessors(center_node))
    hop1_neighbors = set(random.sample(list(hop1_neighbors), min(len(hop1_neighbors), max_nodes)))
    
    hop2_neighbors = set()
    for node in hop1_neighbors:
        hop2_neighbors.update(G.successors(node))
        hop2_neighbors.update(G.predecessors(node))
    
    hop2_neighbors -= hop1_neighbors
    hop2_neighbors.discard(center_node)

    sampled_2hop = random.sample(list(hop2_neighbors), min(len(hop2_neighbors), max_nodes))
    nodes_in_subgraph = set([center_node]) | hop1_neighbors | set(sampled_2hop)
    subG = G.subgraph(nodes_in_subgraph).copy()
    
    if edge_remove:
        edges = list(subG.edges())
        num_edges_to_keep = int(len(edges) * edge_keep_ratio)
        edges_to_keep = set(random.sample(edges, num_edges_to_keep))
        
        for edge in edges:
            if edge not in edges_to_keep:
                subG.remove_edge(*edge)
    return subG




def generate_edge_list(g):
    txt=str(g.edges())[1:-1].replace(', ',' ')
    
    txt=txt.replace(') (',' | ')
    txt=txt.replace('(','')
    txt=txt.replace(')','')
    return txt



def get_sync_graph_data(base_model_path, max_num,method,n,density,pattern_name,edge_remove,edge_remove_density, test_num = 1000):
    counts = 0
    G_full = generate_fully_connected_digraph(n)

    graph_set = []
    G_sub = subgraph_by_density(G_full, density)

    if os.path.exists(base_model_path)==False:
        os.makedirs(base_model_path)

    with open(os.path.join(base_model_path, f'data/{pattern_name}-{density}.pkl'),'wb') as f:
        pickle.dump(G_sub, f)

    print(f"Density={density}, edge_num={G_sub.number_of_edges()}")
    string_length = []
    dataset = []
    max_legnth = 0

    while True:
        nodes_list = list(G_sub.nodes())
        centern_node = random.choices(nodes_list)[0]
        G_small = sample_2hop_subgraph(G_sub, center_node=centern_node, max_nodes=5, edge_keep_ratio=edge_remove_density,edge_remove=edge_remove)
        nodes = list(G_small.nodes())
        sets_nodes = []
        G_txt = generate_edge_list(G_small)
        if G_txt not in graph_set:
            graph_set.append(G_txt)
        else:continue
        for _ in range(int(0.5*len(nodes)*len(nodes)/2)):
            s, e = random.choices(nodes, k = 2)
            if s == e:continue
            if [s, e] in sets_nodes: continue
            try:
                subgraph_txt = '<START> '+ G_txt
                shortest_path = list(nx.all_shortest_paths(G_small, source=s, target=e))
                for path in shortest_path:
                    if len(path) <= 2:continue
                    string = subgraph_txt + f' S {str(s)} E {str(e)} PATH '
                    if len(path) > max_legnth:
                        max_legnth = len(path)
                    for node in path:
                        string += str(node) + ' '
                    string += 'END_P <END>'

                    string_length.append(len(string.split(' ')))
                    dataset.append(string)
                    counts+=1
            except:
                continue
        print('generating:', counts, 'goal:', max_num, 'max path length', max_legnth)
        if counts >= max_num:break
        
    tokenizer = Tokenizer(models.WordLevel(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordLevelTrainer(special_tokens=["<UNK>"])
    tokenizer.train_from_iterator(dataset, trainer)
    tokenizer = PreTrainedTokenizerFast(min_frequency=2,
                                        tokenizer_object=tokenizer)
    tokenizer.add_special_tokens({'pad_token': '<PAD>',
                                'eos_token': '<END>',
                                'unk_token': '<UNK>',
                                'bos_token': '<START>'})
    tokenizer_path = os.path.join(base_model_path, f"{pattern_name}_{method}_baby_tokenizer.json")
    tokenizer.save_pretrained(tokenizer_path)

    test_num = test_num
    train_num = max_num
    valid_num = int(train_num * 0.1)
    valid_corpus = dataset[test_num:valid_num + test_num]
    test_corpus = dataset[:test_num]
    training_corpus = dataset[valid_num + test_num:train_num]
    print('max string length', max(string_length))
    with open(os.path.join(f'{base_model_path}',f'train_{pattern_name}_{density}_{edge_remove_density}_{method}.pkl'),'wb') as f:
        pickle.dump(training_corpus,f)
    with open(os.path.join(f'{base_model_path}',f'valid_{pattern_name}_{density}_{edge_remove_density}_{method}.pkl'),'wb') as f:
        pickle.dump(valid_corpus,f)
    with open(os.path.join(f'{base_model_path}',f'test_{pattern_name}_{density}_{edge_remove_density}_{method}.pkl'),'wb') as f:
        pickle.dump(test_corpus,f)
                
if __name__ == '__main__':
    base_model_path=f'base'
    max_num = 1000000
    method = 'path'
    n = 10
    pattern_name = f'sync{n}'
    edge_remove = True
    edge_remove_density = 0.4
    density = 0.8
    get_sync_graph_data(base_model_path, max_num,method,n,density,pattern_name,edge_remove,edge_remove_density)