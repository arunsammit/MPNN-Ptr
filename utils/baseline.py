from utils.utils import communication_cost, get_reverse_mapping
import torch
from scipy import stats
from math import sqrt
from torch_scatter import scatter

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
def paired_t_test(penalty_curr:torch.Tensor, penalty_baseline:torch.Tensor) -> int:
    # FIXME: complete paired t-test function
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
