from utils.utils import communication_cost, get_reverse_mapping
import torch
from scipy import stats
from math import sqrt
from torch_scatter import scatter
from torch_geometric.loader import DataLoader
from utils.datagenerate import generate_graph_data_list, generate_distance_matrix

@torch.no_grad()
def calculate_baseline(edge_index, edge_attr, batch, batch_size, distance_matrix, samples, num_samples):
    # samples shape: [batch_size, num_samples, seq_len]
    reverse_mappings = get_reverse_mapping(samples)
    reverse_mappings_flattened = reverse_mappings[reverse_mappings != -1]
    # BUG: The averaging is not done considering all the samples 
    edge_index_repeated = edge_index.repeat_interleave(num_samples, dim=1)
    edge_attr_repeated = edge_attr.repeat_interleave(num_samples, dim=0)
    costs = distance_matrix[reverse_mappings_flattened[edge_index_repeated[0]], reverse_mappings_flattened[edge_index_repeated[1]]].unsqueeze(-1)
    comm_cost = costs * edge_attr_repeated
    baseline_each = scatter(comm_cost, batch[edge_index_repeated[0]], dim=0, dim_size=batch_size, reduce='mean')
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
	    communication_cost(data_batched.edge_index, data_batched.edge_attr, data_batched.batch, data_batched.num_graphs, distance_matrix, predicted_mappings_current)
    predicted_mappings_baseline, _ = \
	    mpnn_ptr_baseline(data_batched,1)
    penalty_baseline = \
        communication_cost(data_batched.edge_index, data_batched.edge_attr, data_batched.batch, data_batched.num_graphs, distance_matrix, predicted_mappings_baseline)
    p_value, t_value, mean = paired_t_test(penalty_current, penalty_baseline)
    print(f'p-value: {p_value}, t-value: {t_value}, mean: {mean}')
    if(p_value < 0.05):
        mpnn_ptr_baseline.load_state_dict(mpnn_ptr.state_dict())
if __name__ == '__main__':
    # create a dataloader
    import math
    batch_size = 2
    graph_size = 9
    data_list = generate_graph_data_list(graph_size=graph_size, num_graphs=batch_size)
    data_loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
    num_samples = 2
    predicted_mappings = torch.zeros(num_samples * batch_size, graph_size)
    for i in range(num_samples*batch_size):
        predicted_mappings[i,:] = torch.randperm(graph_size)
    predicted_mappings = predicted_mappings.long()
    data = next(iter(data_loader))
    n = math.ceil(math.sqrt(graph_size))
    m = math.ceil(graph_size/n)
    distance_matrix = generate_distance_matrix(n,m)
    baseline = calculate_baseline(data.edge_index, data.edge_attr, data.batch, batch_size, distance_matrix, predicted_mappings, num_samples)
