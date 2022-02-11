#%%
from collections import defaultdict
import networkx as nx
import random
import numpy as np
from numpy.random import default_rng
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from typing import List
import torch
import sys
from collections import defaultdict
import math
#%%
def generate_graph_data_loader_with_distance_matrix(sizes_list, batch_size, device=torch.device('cpu'), shuffle=False) -> DataLoader:
    n = np.ceil(np.sqrt(sizes_list)).astype(int)
    m = np.ceil(sizes_list/ n).astype(int)
    datalist = []
    distance_matrices = []
    rng = default_rng()
    for i in range(len(sizes_list)):
        D = generate_distance_matrix(n[i], m[i])
        distance_matrices.append(D.to(device))
    max_size = max(sizes_list)
    for size in sizes_list:
        for j in range(batch_size):
            G = nx.generators.random_graphs.fast_gnp_random_graph(size, .3, directed=True)
            uniform = 1 - rng.uniform(0, 1, len(G.edges))
            nx.set_edge_attributes(G, {e: {'weight': uniform[i]} for i, e in enumerate(G.edges)})
            edge_index = torch.tensor(list(G.edges), dtype=torch.long)
            x = torch.from_numpy(nx.to_numpy_array(G)).float()
            x_padded = torch.cat([x, torch.zeros(x.shape[0], max_size - x.shape[1])], dim=1).to(device)
            edge_attr = torch.from_numpy(uniform).float().unsqueeze(-1).to(device)
            data = Data(x=x_padded, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr)
            data = data.to(device)
            datalist.append(data)
    dataloader = DataLoader(datalist, batch_size=batch_size, shuffle=shuffle)
    return dataloader, distance_matrices


def _generate_distance_matrix(n,m, mapping):
    G = nx.generators.lattice.grid_2d_graph(n, m)
    G = nx.relabel_nodes(G, mapping)
    gen = nx.algorithms.shortest_paths.unweighted.all_pairs_shortest_path_length(G)
    D = np.zeros((len(G.nodes), len(G.nodes)))
    for i, d in gen:
        for j, val in d.items():
            D[i, j] = val
    return torch.from_numpy(D)
def generate_distance_matrix(n,m, numbering='default'):
    """
    If numbering is 'default' then for n = 4 and m = 4, numbering is like:
    0  1  2  3  4  
    5  6  7  8  9  
    10 11 12 13 14 
    15 16 17 18 19 

    else if numbering is 'new' then for n = 4 and m = 4, numbering is like:
    0  1  4  9  16 .  .  .
    3  2  5  10 17 .  .  .
    8  7  6  11 18 .  .  .
    15 14 13 12 19 .  .  .
    .  .  .  .  .  .  .  .
    .  .  .  .  .  .  .  .
    .  .  .  .  .  .  .  .
    """
    if numbering == 'default':
        mapping_func = lambda k, l: m * k + l
    elif numbering == 'new':
        mapping_func = lambda k, l: k*(k+2) - l if k > l else l**2 + k
    else:
        raise ValueError('numbering must be either default or new')
    mapping = {(k, l): mapping_func(k, l) for k in range(n) for l in range(m)}    
    return _generate_distance_matrix(n,m, mapping)

class DistanceMatrix(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def __missing__(self, graph_size):
        print(f"Generating distance matrix for graph size {graph_size}")
        n = math.ceil(math.sqrt(graph_size))
        m = math.ceil(graph_size/n)
        value = generate_distance_matrix(n, m)
        self[graph_size] = value
        return value
class DistanceMatrixNew:
    @staticmethod
    def is_valid(num_nodes)->bool:
        n = math.ceil(math.sqrt(num_nodes))
        m = math.ceil(num_nodes / n)
        return n * m == num_nodes and abs(n - m) <= 1
    def __init__(self, max_num_nodes):
        if not self.is_valid(max_num_nodes):
            raise ValueError(f"max_num_nodes must be either a perfect square or product of two consecutive integers")
        n = math.ceil(math.sqrt(max_num_nodes))
        m = math.ceil(max_num_nodes / n)
        self.distance_matrix = generate_distance_matrix(n, m, numbering='new')
        self.max_num_nodes = max_num_nodes

    def __getitem__(self, k:int )->torch.Tensor:
        if not self.is_valid(k):
            raise ValueError(f"key must be either a perfect square or product of two consecutive integers")
        return self.distance_matrix[:k, :k]
        

# %%

def generate_graph_data_list(graph_size: int, num_graphs: int) -> List[Data]:
    rng = default_rng()
    datalist = []
    for i in range(num_graphs):
        G = nx.generators.random_graphs.fast_gnp_random_graph(graph_size, .3, directed=True)
        uniform = 1- rng.uniform(0, 1, len(G.edges))
        nx.set_edge_attributes(G, {e: {'weight': uniform[i]} for i, e in enumerate(G.edges)})
        edge_index = torch.tensor(list(G.edges), dtype=torch.long)
        x = torch.from_numpy(nx.to_numpy_array(G)).float()
        edge_attr = torch.from_numpy(uniform).float().unsqueeze(-1)
        data = Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr)
        datalist.append(data)
    return datalist
# def normalize_graph_data(data:Data)->Data:
#     x_max = data.x.max()
#     data.x = data.x / x_max
#     data.edge_attr = data.edge_attr / x_max
#     return data
    


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # graph_data_list = generate_graph_data_list(min_graph_size, max_graph_size)
    # sizes_list = np.array([9, 12, 16, 20, 25, 30, 36, 49, 64, 81])
    # dataloader49, distance_matrices49 = generate_graph_data_loader_with_distance_matrix(sizes_list, 128, device)
    # torch.save([dataloader49, distance_matrices49], 'data/data_81.pt')
    # num_graphs = 1024
    # batch_size = 128
    # graph_size = 64
    # num_batches = num_graphs // batch_size
    # dataloader49_single, distance_matrices49_single = generate_graph_data_loader_with_distance_matrix(np.array(num_batches * [graph_size]), batch_size, device, shuffle=True)
    # torch.save([dataloader49_single, distance_matrices49_single], 'data/data_single_64.pt')
    if len(sys.argv) < 2:
        print(
        "Usage:\n" + \
            "\tpython3 datagenerate.py single_instance <graph_size>\n" + \
            "\tpython3 datagenerate.py single <graph_size> <num_graphs>\n" + \
            "\tpython3 datagenerate.py multi <graph_sizes (multiple values with space)> <num_batches> <batch_size>"
        )
        sys.exit(1)
    if sys.argv[1] == 'single_instance':
        graph_size = int(sys.argv[2])
        save_path = 'data/data_single_instance_uniform_{}.pt'.format(graph_size)
        single_graph = generate_graph_data_list(graph_size, 1)[0].to(device)
        torch.save(single_graph, save_path)
    elif sys.argv[1] == 'single':
        graph_size = int(sys.argv[2])
        num_graphs = int(sys.argv[3])
        datalist = generate_graph_data_list(graph_size, num_graphs)
        torch.save(datalist, f'data/data_single_{graph_size}_{num_graphs}.pt')
    elif sys.argv[1] == 'multi':
        sizes_list = np.array(sys.argv[2:len(sys.argv)-2])
        num_batches = int(sys.argv[-2])
        batch_size = int(sys.argv[-1])
        max_graph_size = sizes_list.max()
        dataloader, distance_matrices = generate_graph_data_loader_with_distance_matrix(sizes_list, batch_size, device, shuffle=True)
        torch.save([dataloader, distance_matrices], 'data/data_{}.pt'.format(max_graph_size))





# %%
