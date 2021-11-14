import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import torch_geometric.utils
from numpy.random import default_rng
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from typing import Tuple, List
import torch
def generate_graph_data_loader_with_distance_matrix(batch_size):
    sizes_list = np.array([9, 12, 16, 20, 25, 30, 36])
    n = np.ceil(np.sqrt(sizes_list)).astype(int)
    m = np.ceil(sizes_list/ n).astype(int)
    datalist = []
    distance_matrices = []
    for i in range(len(sizes_list)):
        G = nx.generators.lattice.grid_2d_graph(n[i], m[i])
        mapping = {(k, l): m[i]*k + l for k in range(n[i]) for l in range(m[i])}
        G = nx.relabel_nodes(G, mapping)
        gen = nx.algorithms.shortest_paths.unweighted.all_pairs_shortest_path_length(G)
        D = np.zeros((len(G.nodes), len(G.nodes)))
        for i, d in gen:
            for j, val in d.items():
                D[i, j] = val
        distance_matrices.append(torch.from_numpy(D))
    max_size = max(sizes_list)
    for size in sizes_list:
        for j in range(batch_size):
            G = nx.generators.random_graphs.fast_gnp_random_graph(size, .3, directed=True)
            lognormal = np.random.lognormal(1, 3, len(G.edges))
            nx.set_edge_attributes(G, {e: {'weight': lognormal[i]} for i, e in enumerate(G.edges)})
            edge_index = torch.tensor(list(G.edges), dtype=torch.long)
            x = torch.from_numpy(nx.to_numpy_array(G)).float()
            x_padded = torch.cat([x, torch.zeros(x.shape[0], max_size - x.shape[1])], dim=1)
            edge_attr = torch.from_numpy(lognormal).float().unsqueeze(-1)
            data = Data(x=x_padded, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr)
            datalist.append(data)
    dataloader = DataLoader(datalist, batch_size=batch_size)
    return dataloader, distance_matrices

def generate_graph_data_list(min_graph_size=10, max_graph_size=50) -> List[Data]:
    if max_graph_size < min_graph_size:
        raise ValueError("max_graph_size must be greater than min_graph_size")
    rng = default_rng()
    datalist = []
    graph_sizes = random.choices(range(min_graph_size, max_graph_size, 1), k=100)
    for i in range(100):
        G = nx.generators.random_graphs.fast_gnp_random_graph(graph_sizes[i], .3, directed=True)
        lognormal = rng.lognormal(1, 3, len(G.edges))
        nx.set_edge_attributes(G, {e: {'weight': lognormal[i]} for i, e in enumerate(G.edges)})
        edge_index = torch.tensor(list(G.edges), dtype=torch.long)
        x = torch.from_numpy(nx.to_numpy_array(G)).float()
        x_padded = torch.cat([x, torch.zeros(x.shape[0], max_graph_size - x.shape[1])], dim=1)
        edge_attr = torch.from_numpy(lognormal).float().unsqueeze(-1)
        # print(edge_attr.shape)
        data = Data(x=x_padded, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr)
        datalist.append(data)
    return datalist

if __name__ == '__main__':
    # min_graph_size = 5
    # max_graph_size = 10
    # graph_data_list = generate_graph_data_list(min_graph_size, max_graph_size)
    dataloader, distance_matrices =  generate_graph_data_loader_with_distance_matrix(10)
    for data, distance_matrix in zip(dataloader,distance_matrices):
        print(data)
        print(distance_matrix.shape)

