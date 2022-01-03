from utils.utils import communication_cost, get_reverse_mapping
import torch
from scipy import stats
from math import sqrt
from torch_scatter import scatter
from torch_geometric.loader import DataLoader
from utils.datagenerate import generate_graph_data_list, generate_distance_matrix

@torch.no_grad()
def calculate_baseline(edge_index, edge_attr, batch, distance_matrix, samples, num_samples):
    # samples shape: [batch_size * num_samples, seq_len]
    # convert from repeat_interleave to repeat
    graph_size = samples.size(1)
    batch_size = samples.size(0) // num_samples
    device = edge_index.device
    reverse_mappings = get_reverse_mapping(samples)
    reverse_mappings_flattened = reverse_mappings[reverse_mappings != -1]
    edge_index_repeated = edge_index.repeat(1, num_samples)
    edge_index_adjust = \
        torch.arange(num_samples, device = device).repeat_interleave(edge_index.size(1)) * graph_size * batch_size
    edge_index_adjusted = edge_index_repeated + edge_index_adjust
    edge_attr_repeated = edge_attr.repeat(num_samples, 1)
    costs = distance_matrix[reverse_mappings_flattened[edge_index_adjusted[0]], reverse_mappings_flattened[edge_index_adjusted[1]]].unsqueeze(-1)
    comm_cost = costs * edge_attr_repeated
    baseline_each = scatter(comm_cost, batch[edge_index_repeated[0]], dim=0, dim_size=batch_size, reduce='sum').squeeze(-1)
    baseline_each = baseline_each / num_samples
    return baseline_each

@torch.no_grad()
def paired_t_test(penalty_curr:torch.Tensor, penalty_baseline:torch.Tensor) -> int:
    # penalty_curr: [batch_size]
    # penalty_baseline: [batch_size]
    diff = penalty_curr - penalty_baseline
    mean = diff.mean().item()
    # null hypothesis: mean = 0
    # alternative hypothesis: mean < 0 
    std = diff.std(unbiased=True).item()
    t_value = (mean - 0) / (std / (sqrt(penalty_curr.size(0))))
    # calculate p value using scipy
    p_value = stats.t.cdf(t_value, penalty_curr.size(0) - 1)
    return p_value, t_value, mean
@torch.no_grad()
def baseline_model_update(data_batched, distance_matrix, mpnn_ptr, mpnn_ptr_baseline):
    mpnn_ptr.eval()
    mpnn_ptr.decoding_type = 'greedy'
    predicted_mappings_current, _ = \
	    mpnn_ptr(data_batched,1)
    mpnn_ptr.decoding_type ='sampling'
    penalty_current = \
	    communication_cost(data_batched.edge_index, data_batched.edge_attr, data_batched.batch, distance_matrix, predicted_mappings_current)
    predicted_mappings_baseline, _ = \
	    mpnn_ptr_baseline(data_batched,1)
    penalty_baseline = \
        communication_cost(data_batched.edge_index, data_batched.edge_attr, data_batched.batch, distance_matrix, predicted_mappings_baseline)
    p_value, t_value, mean = paired_t_test(penalty_current, penalty_baseline)
    print(f'p-value: {p_value}, t-value: {t_value}, mean: {mean}')
    if(p_value < 0.05):
        mpnn_ptr_baseline.load_state_dict(mpnn_ptr.state_dict())
if __name__ == '__main__':
    # create a dataloader
    import math
    batch_size = 4
    graph_size = 6
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
    baseline = calculate_baseline(data.edge_index, data.edge_attr, data.batch, distance_matrix, predicted_mappings, num_samples)
