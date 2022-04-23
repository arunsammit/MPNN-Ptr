from models.mpnn import Mpnn
from torch import nn
from models.seqToseq import PointerNet
import torch
import torch_geometric
from models.transformers import TransformerPointerNet
from models.transformersV2 import TransformerPointerNet2

# combine Mpnn and PointeNet
class MpnnPtr(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, K, n_layers, p_dropout, device, logit_clipping=True, decoding_type='sampling', feature_scale=1.0):
        # K is number of rounds of message passing
        super(MpnnPtr, self).__init__()
        self.mpnn = Mpnn(input_dim, embedding_dim, K)
        self.device = device
        self.logit_clipping = logit_clipping
        self.feature_scale = feature_scale
        self.ptr_net = PointerNet(embedding_dim, hidden_dim, n_layers, p_dropout, device, logit_clipping, decoding_type)
    @property
    def decoding_type(self):
        return self.ptr_net.decoding_type
    @decoding_type.setter
    def decoding_type(self, decoding_type):
        self.ptr_net.decoding_type = decoding_type
    def forward(self, data, num_samples=1):
        # data is batch of graphs
        # pass data through Mpnn to get embeddings
        embeddings = self.mpnn(data.x / self.feature_scale, data.edge_index, data.edge_attr / self.feature_scale, data.batch)
        batched_embeddings, mask = torch_geometric.utils.to_dense_batch(embeddings, data.batch)
        # batched_embeddings shape: (batch_size, max_num_nodes, embedding_dim)
        # pass embeddings and mask through PointerNet to get pointer
        return self.ptr_net(batched_embeddings.permute(1, 0, 2), mask, num_samples)
class MpnnTransformer(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, K, n_layers, p_dropout, device, logit_clipping=True, decoding_type='sampling', feature_scale=1.0, version='v1'):
        # K is the number of rounds of message passing
        super(MpnnTransformer, self).__init__()
        # self.mpnn = Mpnn(input_dim, embedding_dim, K)
        self.device = device
        self.logit_clipping = logit_clipping
        self.feature_scale = feature_scale
        if version == 'v1':
            self.t_ptr_net = TransformerPointerNet(embedding_dim, hidden_dim, n_layers, p_dropout, device, logit_clipping, decoding_type=decoding_type)
        elif version == 'v2':
            self.t_ptr_net = TransformerPointerNet2(embedding_dim, hidden_dim, n_layers, p_dropout, device, logit_clipping, decoding_type=decoding_type)
    @property
    def decoding_type(self):
        return self.t_ptr_net.decoding_type
    @decoding_type.setter
    def decoding_type(self, decoding_type):
        self.t_ptr_net.decoding_type = decoding_type
    def forward(self, data, num_samples=1):
        # data is batch of graphs
        # pass data through Mpnn to get embeddings
        embeddings = self.mpnn(data.x / self.feature_scale, data.edge_index, data.edge_attr / self.feature_scale, data.batch)
        batched_embeddings, mask = torch_geometric.utils.to_dense_batch(embeddings, data.batch)
        # batched_embeddings shape: (batch_size, max_num_nodes, embedding_dim)
        # pass embeddings and mask through PointerNet to get pointer
        return self.t_ptr_net(batched_embeddings.permute(1, 0, 2), mask, num_samples)
def main():
    from utils.datagenerate import generate_graph_data_list
    from utils.datagenerate import generate_graph_data_loader_with_distance_matrix
    from torch_geometric.loader import DataLoader

    # min_graph_size = 5
    # graph_data_list = generate_graph_data_list(min_graph_size, max_graph_size)
    # data_loader = DataLoader(graph_data_list, batch_size=4)
    # use generate_graph_data_loader_with_distance_matrix
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader, distance_matrices  = generate_graph_data_loader_with_distance_matrix([9,12,16,20,36],10)
    max_graph_size = 36
    mpnn_ptr = MpnnPtr(input_dim=max_graph_size, embedding_dim=50, hidden_dim=50, K=2, n_layers=2, p_dropout=0.1, device=device)
    for data, distance_matrix in zip(dataloader, distance_matrices):
        predicted_mappings, log_likelihoods_sum = mpnn_ptr(data)
        print(predicted_mappings)
        print(log_likelihoods_sum)
if __name__ == '__main__':
    main()
