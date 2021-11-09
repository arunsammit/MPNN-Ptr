# %%
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
from numpy.random import default_rng
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from typing import Tuple, List
import torch


def generate_data_loader(min_graph_size=10, max_graph_size=50) -> List[Data]:
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
# %%
