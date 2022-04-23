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

def get_mesh_dimensions_newer(num_nodes):
    """
    returns first_dim, second_dim for a 2D mesh suitable for given number of nodes according to newer method of numbering routers
    """
    n = math.floor(math.sqrt(num_nodes))
    m = n
    if n * m != num_nodes:
        if n & 1:
            n += 1
        else:
            m += 1
    if n * m != num_nodes:
        raise ValueError(f"max_num_nodes must be either a perfect square or product of two consecutive integers")
    return n, m

def _generate_distance_matrix(dims, mapping):
    G = nx.grid_graph(dims)
    G = nx.relabel_nodes(G, mapping)
    gen = nx.algorithms.shortest_paths.unweighted.all_pairs_shortest_path_length(G)
    D = np.zeros((len(G.nodes), len(G.nodes)))
    for i, d in gen:
        for j, val in d.items():
            D[i, j] = val
    return torch.from_numpy(D)
def num_from_coord(x ,y):

    if x >= y:
        if x & 1: # x is odd
            return x**2 + y
        else: # x is even
            return x**2 +2*x - y
    else:
        if not(y & 1): # y is even
            return y**2 + x
        else: # y is odd
            return y**2 + 2*y - x
            
class NocNumbering():
    @staticmethod
    def __init__(self, mesh):
        self.mesh = mesh
        self.num_to_coord = {num: (i, j) for i, row in enumerate(mesh) for j, num in enumerate(row)}
        self.adj_idxs = {num: self.compute_adj_idx(num) for num in set(self.num_to_coord) - {0, 1}}
    def get_num(self, x, y):
        return self.mesh[x, y]
    def get_coord(self, num):
        return self.num_to_coord[num]
    def get_adj_idx(self, num):
        if num == 0 or num == 1 or num >= self.mesh.shape[0] * self.mesh.shape[1]:
            raise ValueError(f"{num} is not a valid node number")
        return self.adj_idxs[num]
    def get_neigh(self, num, max_num=None, diag=False):
        x, y = self.get_coord(num)
        if max_num is not None:
            max_x, max_y = self.get_coord(max_num)
        else:
            max_x, max_y = self.mesh.shape
        neigh = set()
        if x > 0:
            neigh.add(self.get_num(x-1, y))
        if x < max_x - 1:
            neigh.add(self.get_num(x+1, y))
        if y > 0:
            neigh.add(self.get_num(x, y-1))
        if y < max_y - 1:
            neigh.add(self.get_num(x, y+1))
        if diag:
            if x > 0 and y > 0:
                neigh.add(self.get_num(x-1, y-1))
            if x > 0 and y < max_y - 1:
                neigh.add(self.get_num(x-1, y+1))
            if x < max_x - 1 and y > 0:
                neigh.add(self.get_num(x+1, y-1))
            if x < max_x - 1 and y < max_y - 1:
                neigh.add(self.get_num(x+1, y+1))
        return neigh
    def compute_adj_idx(self, num):
        def cleanup(adj_idxs):
            adj_idxs.discard(num-1)
            for val in list(adj_idxs):
                if val > num:
                    adj_idxs.discard(val)    
        adj_idxs = self.get_neigh(num)    
        cleanup(adj_idxs)
        if len(adj_idxs) == 0:
            adj_idxs = self.get_neigh(num, diag=True)
            cleanup(adj_idxs)
        if len(adj_idxs) != 1:
            print(f"{num} has {len(adj_idxs)} neighbors")
            print(self.mesh)
            raise ValueError('adj_idxs should have length 1, probably there is a bug somewhere')
        return adj_idxs.pop()

    
class NocNumberingNew(NocNumbering):
    def __init__(self, max_num_nodes = 121):
        d1, d2 = get_mesh_dimensions_newer(max_num_nodes)
        mesh = np.full((d1, d2), -1, dtype=int)
        for i in range(mesh.shape[0]):
            for j in range(mesh.shape[1]):
                mesh[i, j] = num_from_coord(i, j)
        super().__init__(self, mesh)
def generate_distance_matrix(n,m, numbering='default'):
    """
    If numbering is 'default' then for n = 4 and m = 4, numbering is like:
    0  1  2  3  4  
    5  6  7  8  9  
    10 11 12 13 14 
    15 16 17 18 19 

    else if numbering is 'new' then for n = 4 and m = 4, numbering is like:
    0  3  4  15 16  .  .  .
    1  2  5  14 17 .  .  .
    8  7  6  13 18 .  .  .
    9  10 11 12 19 .  .  .
    .  .  .  .  .  .  .  .
    .  .  .  .  .  .  .  .
    .  .  .  .  .  .  .  .
    """
    if numbering == 'default':
        mapping_func = lambda k, l: m * k + l
    elif numbering == 'new':
        print("Using newer numbering")
        mapping_func = num_from_coord
    else:
        raise ValueError('numbering must be either default or new')
    mapping = {(k, l): mapping_func(k, l) for k in range(n) for l in range(m)}    
    return _generate_distance_matrix((m,n), mapping)
def generate_distance_matrix_3D(n, m, l):
    mapping = {(i, j, k): i + j*n + k*m*n for i in range(n) for j in range(m) for k in range(l)}
    return _generate_distance_matrix((l, m, n), mapping)

class DistanceMatrix(dict):
    def __init__(self, *args, **kwargs):
        print("Using older sequence of decoding")
        super().__init__(*args, **kwargs)
    def __missing__(self, graph_size):
        # print(f"Generating distance matrix for graph size {graph_size}")
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
        print("using newer sequence of decoding")
        n, m = get_mesh_dimensions_newer(max_num_nodes)
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
