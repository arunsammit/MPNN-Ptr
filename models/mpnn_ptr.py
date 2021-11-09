from mpnn import Mpnn
from torch import nn
from seqToseq import PointerNet
import torch
import torch_geometric
# combine Mpnn and PointeNet
class MpnnPtr(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, K, n_layers, p_dropout):
        # K is number of rounds of message passing
        super(MpnnPtr, self).__init__()
        self.mpnn = Mpnn(input_dim, embedding_dim, K)
        self.ptr_net = PointerNet(embedding_dim, hidden_dim, n_layers, p_dropout)
    def forward(self, data):
        # data is batch of graphs
        # pass data through Mpnn to get embeddings
        embeddings = self.mpnn(data.x, data.edge_index, data.edge_attr, data.batch)
        # convert the embeddings to pack_padded_sequence
        batched_embeddings, mask = torch_geometric.utils.to_dense_batch(embeddings, data.batch)
        # batched_embeddings shape: (batch_size, max_num_nodes, embedding_dim)
        # pass embeddings and mask through PointerNet to get pointer
        pointer = self.ptr_net(batched_embeddings.permute(1, 0, 2), mask)




