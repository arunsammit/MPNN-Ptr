import torch
from torch import nn
from torch import Tensor
from models.mpnn import Mpnn, ConnectionsEmbedding
from data.datagenerate import generate_data_loader
from torch_geometric.loader import DataLoader
# %%
# testing ConnectionsEmbedding
max_graph_size = 50
datalist = generate_data_loader(max_graph_size=max_graph_size)
dataloader = DataLoader(datalist, batch_size=4, shuffle=True)
connections_embedding = ConnectionsEmbedding(in_channels=max_graph_size, n_features=10)
print(datalist[0].x.shape, datalist[0].edge_index.shape, datalist[0].edge_attr.shape)
embeddings = connections_embedding(datalist[0].x, datalist[0].edge_index, datalist[0].edge_attr)
print(embeddings.shape)
# %%
# testing Mpnn
mpnn = Mpnn(in_channels=max_graph_size, n_features=10, K=3)
print(datalist[0].x.shape, datalist[0].edge_index.shape, datalist[0].edge_attr.shape)
embeddings = mpnn(datalist[0].x, datalist[0].edge_index, datalist[0].edge_attr)
print(embeddings.shape)


