# %%
import random
import networkx as nx
import torch
from torch import nn
from numpy.random import default_rng
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from typing import Tuple, Union, Optional
from torch import Tensor

# %%
# Generating synthetic data
rng = default_rng()
datalist = []
graph_sizes = random.choices(range(10, 50, 2), k=100)
max_graph_size = max(graph_sizes)
# %%
for i in range(100):
    G = nx.generators.random_graphs.fast_gnp_random_graph(graph_sizes[i], .3, directed=True)
    lognormal = rng.lognormal(1, 3, len(G.edges))
    nx.set_edge_attributes(G, {e: {'weight': lognormal[i]} for i, e in enumerate(G.edges)})
    edge_index = torch.tensor(list(G.edges), dtype=torch.long)
    # print(edge_index.shape)
    x = torch.from_numpy(nx.to_numpy_array(G)).float()
    # print(x.shape)
    x_padded = torch.cat([x, torch.zeros(x.shape[0], max_graph_size - x.shape[1])], dim=1)
    # print(x_padded.shape)
    # print(x_padded.shape)
    edge_attr = torch.from_numpy(lognormal).float().unsqueeze(-1)
    print(edge_attr.shape)
    data = Data(x=x_padded, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr)
    datalist.append(data)
loader = DataLoader(datalist, batch_size=4, shuffle=True)
# %%
cnt = 0
for data in datalist:
    print(data)
    cnt += 1
    if cnt == 4:
        break
# cnt = 0
for batch in loader:
    print(batch)
    # cnt += 1
    break
print(cnt)


# %%
class ConnectionsEmbedding(MessagePassing):
    def __init__(self, in_channels, n_features):
        super(ConnectionsEmbedding, self).__init__(aggr='mean')
        self.layer2 = nn.Sequential(
            nn.Linear(in_channels + 1, n_features - 1, bias=False),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(n_features, n_features, bias=False),
            nn.ReLU()
        )

    def message(self, x_j, edge_attr):
        # print(x_j.shape, edge_attr.shape)
        return self.layer2(torch.cat([edge_attr, x_j], dim=1))

    def update(self, aggr_out: Tensor, edge_index: Tensor) -> Tensor:
        # Concatenate node degree to node features.
        # x: [N, in_channels]
        # edge_index: [2, E]
        print(aggr_out.shape, edge_index.shape)
        row, col = edge_index
        deg = degree(col, aggr_out.size(0), dtype=aggr_out.dtype)
        return self.layer3(torch.concat([aggr_out, deg.unsqueeze(-1)], dim=1))

    def forward(self, x, edge_index, edge_attr):
        # x: [N, in_channels]
        # edge_index: [2, E]
        # edge_attr: [E, 1]
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)


# %%
# testing ConnectionsEmbedding
connections_embedding = ConnectionsEmbedding(in_channels=max_graph_size, n_features=10)
print(datalist[0].x.shape, datalist[0].edge_index.shape, datalist[0].edge_attr.shape)
embeddings = connections_embedding(datalist[0].x, datalist[0].edge_index, datalist[0].edge_attr)
print(embeddings.shape)


# %%
# Message Passing Main Layer
class Mpnn(MessagePassing):
    def __init__(self, in_channels, n_features, K):
        super().__init__(aggr='add')
        self.K = K
        self.node_init_embedding_layer = nn.Sequential(
            nn.Linear(in_channels, n_features, bias=False),
            nn.ReLU()
        )
        self.theta4layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * n_features, n_features, bias=False),
                nn.ReLU()
            )
            for _ in range(K)
        ]
        )
        self.theta5layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * n_features, n_features, bias=False),
                nn.ReLU()
            )
            for _ in range(K)
        ]
        )
        self.theta6 = nn.Sequential(
            nn.Linear(n_features, n_features, bias=False),
            nn.ReLU()
        )
        self.theta7 = nn.Linear(2 * n_features, n_features, bias=False)
        self.connection_embedding_layer = ConnectionsEmbedding(in_channels=in_channels, n_features=n_features)

    def forward(self, x, edge_index, edge_attr):
        connections_embedding = self.connection_embedding_layer(x, edge_index, edge_attr)
        x = self.node_init_embedding_layer(x)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        norm = deg.pow(-0.5)
        norm[norm == float('inf')] = 0
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, edge_attr=edge_attr, connections_embedding=connections_embedding,
                               norm=norm, k=k)
        avg_x = x.mean(dim=0)
        avg_x_theta6 = self.theta6(avg_x.unsqueeze(0)).expand(x.shape[0], -1)
        concat = torch.cat([avg_x_theta6, x], dim=1)
        return self.theta7(concat)

    def message(self, x_j, edge_attr):
        # x_j: [E, n_features]
        # edge_attr: [E, 1]
        return edge_attr * x_j

    def update(self, aggr_out, norm, connections_embedding, x, k):
        out1 = aggr_out * norm.view(-1, 1)
        out2 = torch.concat([out1, connections_embedding], dim=1)
        out3 = self.theta4layers[k](out2)
        out4 = self.theta5layers[k](torch.concat([x, out3], dim=1))
        return out4


# %%
# testing Mpnn
mpnn = Mpnn(in_channels=max_graph_size, n_features=10, K=3)
print(datalist[0].x.shape, datalist[0].edge_index.shape, datalist[0].edge_attr.shape)
embeddings = mpnn(datalist[0].x, datalist[0].edge_index, datalist[0].edge_attr)
print(embeddings.shape)
