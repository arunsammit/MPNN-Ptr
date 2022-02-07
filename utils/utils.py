#%%
import torch
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
    # print(f"costs.device: {costs.device}, edge_attr.device: {edge_attr.device}")
    comm_cost = costs * edge_attr
    comm_cost.squeeze_(-1)
    comm_cost_each = scatter(comm_cost, batch[edge_index[0]], dim=0, dim_size=batch_size, reduce='sum')
    return comm_cost_each

@torch.no_grad()
def communication_cost_multiple_samples(edge_index:torch.Tensor, edge_attr, batch, distance_matrix, predicted_mappings, num_samples, calculate_baseline = False):
    """
    Samples should be present in the graph wise fashion. This means that the sample corresponding to the first graph is present at the indices 0, 0 + batch_size, 0 + 2 * batch_size etc. The sample corresponding to the second graph is present at indices 1, 1 + batch_size, 1 + 2 * batch_size etc.
    """
    graph_size = predicted_mappings.size(1)
    batch_size = predicted_mappings.size(0) // num_samples
    device = edge_index.device
    reverse_mappings = get_reverse_mapping(predicted_mappings)
    reverse_mappings_flattened = reverse_mappings[reverse_mappings != -1]
    edge_index_repeated = edge_index.repeat(1, num_samples)
    edge_index_adjust = \
        torch.arange(num_samples, device = device).repeat_interleave(edge_index.size(1)) * graph_size * batch_size
    edge_index_adjusted = edge_index_repeated + edge_index_adjust
    edge_attr_repeated = edge_attr.repeat(num_samples, 1)
    costs = distance_matrix[reverse_mappings_flattened[edge_index_adjusted[0]], reverse_mappings_flattened[edge_index_adjusted[1]]].unsqueeze(-1)
    comm_cost = costs * edge_attr_repeated
    batch_repeated = batch.repeat(num_samples)
    batch_adjust = torch.arange(num_samples, device = device).repeat_interleave(batch.size(0)) * batch_size
    batch_adjusted = batch_repeated + batch_adjust
    comm_cost_each = \
        scatter(comm_cost, batch_adjusted[edge_index_adjusted[0]], dim=0, dim_size = num_samples * batch_size, reduce='sum').squeeze(-1)
    if calculate_baseline:
        indices = torch.arange(batch_size, device = device).repeat(num_samples)
        baseline = scatter(comm_cost_each, indices, dim=0, dim_size = batch_size, reduce='sum') / num_samples
        return comm_cost_each, baseline
    else:
        return comm_cost_each
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
    # create a dataloader
    import math
    batch_size = 4
    graph_size = 6
    from utils.datagenerate import generate_graph_data_list, generate_distance_matrix
    from torch_geometric.loader import DataLoader
    data_list = generate_graph_data_list(graph_size=graph_size, num_graphs=batch_size)
    data_loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
    num_samples = 3
    predicted_mappings = torch.zeros(num_samples * batch_size, graph_size)
    for i in range(num_samples*batch_size):
        predicted_mappings[i,:] = torch.randperm(graph_size)
    predicted_mappings = predicted_mappings.long()
    data = next(iter(data_loader))
    n = math.ceil(math.sqrt(graph_size))
    m = math.ceil(graph_size/n)
    distance_matrix = generate_distance_matrix(n,m)
    baseline = communication_cost_multiple_samples(data.edge_index, data.edge_attr, data.batch, distance_matrix, predicted_mappings, num_samples)

