#%%
import torch
import torch_geometric
from torch_scatter import scatter
from torch import nn
#%%
@torch.no_grad()
def communication_cost(edge_index:torch.Tensor, edge_attr, batch, batch_size, distance_matrix, predicted_mappings):
    reverse_mappings = get_reverse_mapping(predicted_mappings)
    reverse_mappings_flattened = reverse_mappings[reverse_mappings != -1]
    costs = distance_matrix[reverse_mappings_flattened[edge_index[0]], reverse_mappings_flattened[edge_index[1]]]\
        .unsqueeze(-1)
    comm_cost = costs * edge_attr
    comm_cost.squeeze_(-1)
    comm_cost_each = scatter(comm_cost, batch[edge_index[0]], dim=0, dim_size=batch_size, reduce='sum')
    return comm_cost_each
#%%
@torch.no_grad()
def calculate_baseline(edge_index, edge_attr, batch, batch_size, distance_matrix, samples):
    # samples shape: [batch_size, num_samples, seq_len]
    reverse_mappings = get_reverse_mapping(samples.view(-1, samples.size(-1)))
    reverse_mappings_flattened = reverse_mappings[reverse_mappings != -1]
    edge_index_repeated = edge_index.repeat_interleave(samples.size(1), dim=1)
    edge_attr_repeated = edge_attr.repeat_interleave(samples.size(1), dim=0)
    costs = distance_matrix[reverse_mappings_flattened[edge_index_repeated[0]], reverse_mappings_flattened[edge_index_repeated[1]]].unsqueeze(-1)
    comm_cost = costs * edge_attr_repeated
    baseline_each = scatter(comm_cost, batch[edge_index_repeated[0]], dim=0, dim_size=batch_size, reduce='mean')
    return baseline_each
@torch.no_grad()
def paired_t_test(penalty_curr, penalty_baseline):
    # FIXME: complete paired t-test function
    # penalty_curr: [batch_size]
    # penalty_baseline: [batch_size]
    diff:torch.Tensor = penalty_curr - penalty_baseline
    mean = diff.mean()
    std = diff.std(unbiased=True)
    t_value = (mean - 0) / (std / torch.sqrt(penalty_curr.size(0)))
    p_value = 1 - torch.distributions.t.cdf(t_value, penalty_curr.size(0) - 1)
#%%
@torch.no_grad()
def get_reverse_mapping(predicted_mappings):
    device = predicted_mappings.device
    mask = predicted_mappings == -1
    reverse_mappings = torch.zeros_like(predicted_mappings)
    indices = torch.arange(predicted_mappings.size(1)).expand(predicted_mappings.size(0), predicted_mappings.size(1))\
        .clone().to(device)
    # since -1 is not a valid index and predicted mappings are used as indices in scatter function, we need to replace -1 with the valid indices
    predicted_mappings[mask] = indices[mask]
    indices.masked_fill_(mask, -1)
    reverse_mappings.scatter_(1, predicted_mappings, indices)
    return reverse_mappings
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
