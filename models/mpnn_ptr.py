from models.mpnn import Mpnn
from torch import nn
from models.seqToseq import PointerNet
import torch
import torch_geometric

# combine Mpnn and PointeNet
class MpnnPtr(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, K, n_layers, p_dropout):
        # K is number of rounds of message passing
        super(MpnnPtr, self).__init__()
        self.mpnn = Mpnn(input_dim, embedding_dim, K)
        self.ptr_net = PointerNet(embedding_dim, hidden_dim, n_layers, p_dropout)

    def forward(self, data, num_samples=1):
        # data is batch of graphs
        # pass data through Mpnn to get embeddings
        embeddings = self.mpnn(data.x, data.edge_index, data.edge_attr)
        # convert the embeddings to pack_padded_sequence
        batched_embeddings, mask = torch_geometric.utils.to_dense_batch(embeddings, data.batch)
        # batched_embeddings shape: (batch_size, max_num_nodes, embedding_dim)
        # pass embeddings and mask through PointerNet to get pointer
        predicted_mappings, log_likelihoods_sum = self.ptr_net(batched_embeddings.permute(1, 0, 2), mask)
        if num_samples == 1:
            return predicted_mappings, log_likelihoods_sum
        else:
            samples = self.ptr_net.sample_multiple_mappings(batched_embeddings.permute(1,0,2),mask,num_samples)
            return samples, predicted_mappings, log_likelihoods_sum
    def predict_optimal_mappings_greedy(self, data):
        embeddings = self.mpnn(data.x, data.edge_index, data.edge_attr)
        batched_embeddings, mask = torch_geometric.utils.to_dense_batch(embeddings, data.batch)
        predicted_mappings, log_likelihoods_sum = self.ptr_net.predict_optimal_mappings_greedy(batched_embeddings.permute(1, 0, 2), mask)


if __name__ == '__main__':
    from utils.datagenerate import generate_graph_data_list
    from utils.datagenerate import generate_graph_data_loader_with_distance_matrix
    from torch_geometric.loader import DataLoader

    # min_graph_size = 5
    # graph_data_list = generate_graph_data_list(min_graph_size, max_graph_size)
    # data_loader = DataLoader(graph_data_list, batch_size=4)
    # use generate_graph_data_loader_with_distance_matrix
    dataloader, distance_matrices  = generate_graph_data_loader_with_distance_matrix(10)
    max_graph_size = 36
    mpnn_ptr = MpnnPtr(input_dim=max_graph_size, embedding_dim=50, hidden_dim=50, K=2, n_layers=2, p_dropout=0.1)
    for data, distance_matrix in zip(dataloader, distance_matrices):
        predicted_mappings, log_likelihoods_sum = mpnn_ptr(data)
        print(predicted_mappings)
        print(log_likelihoods_sum)
