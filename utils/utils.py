import torch
import torch_geometric
def communication_cost(edge_index:torch.Tensor, edge_attr, distance_matrix, predicted_mappings):
    reverse_mappings = get_reverse_mapping(predicted_mappings)
    reverse_mappings_flattened = reverse_mappings[reverse_mappings != -1]
    costs = distance_matrix[reverse_mappings_flattened[edge_index[0]], reverse_mappings_flattened[edge_index[1]]].unsqueeze(-1)
    comm_cost = costs * edge_attr
    return comm_cost.sum()


def get_reverse_mapping(predicted_mappings):
    mask = predicted_mappings == -1
    reverse_mappings = torch.zeros_like(predicted_mappings)
    indices = torch.arange(predicted_mappings.size(1)).expand(predicted_mappings.size(0), predicted_mappings.size(1)).clone()
    predicted_mappings[mask] = indices[mask]
    indices.masked_fill_(mask, -1)
    reverse_mappings.scatter_(1, predicted_mappings, indices)
    return reverse_mappings
if __name__ == '__main__':
    predicted_mappings = torch.tensor([[0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                       [1, 0, -1, -1, -1, -1, -1, -1, -1, -1],
                                       [2, 1, 0, -1, -1, -1, -1, -1, -1, -1],
                                       [5, 1, 3, 2, 8, 4, 9, 6, 0, 7]])
    # reverse_mappings = get_reverse_mapping(predicted_mappings)
    # print(reverse_mappings)
    from datagenerate import generate_graph_data_loader_with_distance_matrix
    dataloader, distance_matrices =  generate_graph_data_loader_with_distance_matrix(10)
    for data, distance_matrix in zip(dataloader,distance_matrices):

        predicted_mappings = torch.arange(data.num_nodes/data.num_graphs).expand(data.num_graphs, int(data.num_nodes/data.num_graphs)).long()
        print(communication_cost(data.edge_index, data.edge_attr, data.batch, torch.tensor(distance_matrix), predicted_mappings))
