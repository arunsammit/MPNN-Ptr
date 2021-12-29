#%%
import torch
import torch_geometric
from torch_scatter import scatter
from torch import nn

#%%
@torch.no_grad()
def communication_cost(edge_index, edge_attr, batch, distance_matrix, predicted_mappings):
    batch_size = predicted_mappings.size(0) 
    reverse_mappings = get_reverse_mapping(predicted_mappings)
    reverse_mappings_flattened = reverse_mappings[reverse_mappings != -1]
    costs = distance_matrix[reverse_mappings_flattened[edge_index[0]], reverse_mappings_flattened[edge_index[1]]]\
        .unsqueeze(-1)
    comm_cost = costs * edge_attr
    comm_cost.squeeze_(-1)
    comm_cost_each = scatter(comm_cost, batch[edge_index[0]], dim=0, dim_size=batch_size, reduce='sum')
    return comm_cost_each

@torch.no_grad()
def communication_cost_multiple_samples(edge_index:torch.Tensor, edge_attr, batch, distance_matrix, predicted_mappings, num_samples):
    # TODO: This is not correct.
    graph_size = predicted_mappings.size(1)
    batch_size = predicted_mappings.size(0) // num_samples
    reverse_mappings = get_reverse_mapping(predicted_mappings)
    reverse_mappings_flattened = reverse_mappings[reverse_mappings != -1]
    edge_index_repeated = edge_index.repeat(1, num_samples)
    edge_index_adjust = torch.arange(0, reverse_mappings_flattened.size(0), batch_size * graph_size) \
        .repeat_interleave(edge_index.size(1)).expand(2,-1).to(edge_index.device)
    edge_index_adjusted = edge_index_repeated + edge_index_adjust
    edge_attr_repeated = edge_attr.repeat(num_samples, 1)
    costs = distance_matrix[reverse_mappings_flattened[edge_index_adjusted[0]], reverse_mappings_flattened[edge_index_adjusted[1]]].unsqueeze(-1)
    comm_cost = costs * edge_attr_repeated
    comm_cost_each = scatter(comm_cost, batch[edge_index[0]], dim=0, dim_size=batch_size, reduce='sum')
    return comm_cost.squeeze(-1)
#%%
@torch.no_grad()
def get_reverse_mapping(predicted_mappings):
    device = predicted_mappings.device
    mask = predicted_mappings == -1
    reverse_mappings = torch.zeros_like(predicted_mappings)
    indices = torch.arange(predicted_mappings.size(1)).expand(predicted_mappings.size(0), -1)\
        .clone().to(device)
    # since -1 is not a valid index and predicted mappings are used as indices in scatter function, we need to replace -1 with the valid indices
    predicted_mappings[mask] = indices[mask]
    indices.masked_fill_(mask, -1)
    reverse_mappings.scatter_(1, predicted_mappings, indices)
    return reverse_mappings
#%%
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
#%%
if __name__ == '__main__':
    predicted_mappings = torch.tensor([[0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                       [1, 0, -1, -1, -1, -1, -1, -1, -1, -1],
                                       [2, 1, 0, -1, -1, -1, -1, -1, -1, -1],
                                       [5, 1, 3, 2, 8, 4, 9, 6, 0, 7]])
    # reverse_mappings = get_reverse_mapping(predicted_mappings)
    # print(reverse_mappings)
    from datagenerate import generate_graph_data_loader_with_distance_matrix
    import numpy as np
    dataloader, distance_matrices =  generate_graph_data_loader_with_distance_matrix(np.array([9, 12, 16, 20, 25, 30, 36, 49]),100)
    for data, distance_matrix in zip(dataloader,distance_matrices):

        predicted_mappings = torch.arange(data.num_nodes/data.num_graphs).expand(data.num_graphs, int(data.num_nodes/data.num_graphs)).long()
        print(communication_cost(data.edge_index, data.edge_attr, data.batch, data.num_graphs, distance_matrix, predicted_mappings))
