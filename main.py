import torch
from torch import nn
from torch import Tensor
from models.mpnn import Mpnn, ConnectionsEmbedding
from utils.datagenerate import generate_graph_data_list
from torch_geometric.loader import DataLoader
import torch_geometric
# %%
# testing ConnectionsEmbedding
max_graph_size = 50
datalist = generate_graph_data_list(max_graph_size=max_graph_size)
dataloader = DataLoader(datalist, batch_size=4, shuffle=True)
connections_embedding = ConnectionsEmbedding(in_channels=max_graph_size, n_features=10)
# print(datalist[0].x.shape, datalist[0].edge_index.shape, datalist[0].edge_attr.shape)
embeddings = connections_embedding(datalist[0].x, datalist[0].edge_index, datalist[0].edge_attr)
# print(embeddings.shape)

# %%
mpnn = Mpnn(input_node_dim=max_graph_size, n_features=10, K=3)
for data in dataloader:
    print(data)
    embeddings = mpnn(data.x, data.edge_index, data.edge_attr)
    batched_embeddings, mask = torch_geometric.utils.to_dense_batch(embeddings, data.batch,max_num_nodes=max_graph_size)
    batched_embeddings = batched_embeddings.permute(1,0,2)
    print(batched_embeddings.shape)
    break




