# %%
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
from numpy.random import default_rng
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch
# %%
# Generating synthetic data
rng = default_rng()
datalist = []
graph_sizes = random.choices(range(10, 50, 2), k=100)
for i in range(100):
    G = nx.generators.random_graphs.fast_gnp_random_graph(graph_sizes[i], .3, directed=True)
    edge_index = torch.Tensor(list(G.edges))
    edge_attr = torch.from_numpy(rng.lognormal(1,3,len(G.edges)))
    data = Data(edge_index=edge_index,edge_attr=edge_attr)
    datalist.append(data)
loader = DataLoader(datalist, batch_size=32, shuffle=True)

# %%
# Message Passing Neural Network Model



np.save('./graphs.npy', G)
# %%
