import torch
from utils.utils import communication_cost_multiple_samples
from torch_geometric.data import Data
def validate_dataloader(model, dataloader:Data, distance_matrix_dict, beam_width):
    comm_cost = 0
    for data in dataloader:
        data = data.to(model.device)
        distance_matrix = distance_matrix_dict[data.num_nodes]
        _, comm_cost_batch = beam_search_data(model, data, distance_matrix, beam_width)
        comm_cost += float(comm_cost_batch.sum())
    return comm_cost / len(dataloader.dataset)
def beam_search_data(model, data, distance_matrix, beam_width):
    model.eval()
    model.decoding_type = 'greedy'
    with torch.no_grad():
        mappings, _ = model(data, beam_width)
        comm_cost = communication_cost_multiple_samples(data.edge_index, data.edge_attr, data.batch, distance_matrix, mappings, beam_width)
        comm_cost, min_indices = comm_cost.view(beam_width, -1).min(dim=0)
        choosen_mappings = mappings.view(beam_width, data.num_graphs, -1)[min_indices, torch.arange(data.num_graphs)]
    return choosen_mappings, comm_cost