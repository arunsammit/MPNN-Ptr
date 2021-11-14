if __name__ == '__main__':
    from models.mpnn_ptr import MpnnPtr
    from utils.datagenerate import generate_graph_data_loader_with_distance_matrix
    from utils.utils import communication_cost, calculate_baseline
    import torch
    from torch import nn
    dataloader, distance_matrices = generate_graph_data_loader_with_distance_matrix(10)
    # max_graph_size is set to be 36 inside generate_graph_data_loader_with_distance_matrix
    max_graph_size = 36
    mpnn_ptr = MpnnPtr(input_dim=max_graph_size, embedding_dim=50, hidden_dim=50, K=2, n_layers=2, p_dropout=0.1)
    optim = torch.optim.Adam(mpnn_ptr.parameters(), lr=0.0001)
    num_epochs = 100
    for epoch in range(num_epochs):
        for data, distance_matrix in zip(dataloader, distance_matrices):
            num_samples = 5
            samples ,predicted_mappings, log_likelihoods_sum = mpnn_ptr(data,num_samples)
            # samples shape: (batch_size, num_samples, max_graph_size_in_batch)
            # predicted_mappings shape: (batch_size, max_graph_size_in_batch)
            # log_likelihoods_sum shape: (batch_size,)
            reward = - communication_cost(data.edge_index, data.edge_attr, data.batch, data.num_graphs, distance_matrix, predicted_mappings)
            baselines = calculate_baseline(data.edge_index, data.edge_attr, data.batch, data.num_graphs, distance_matrix, samples)
            loss = (1/data.num_graphs) * torch.sum((reward.detach() - baselines.detach()) * log_likelihoods_sum)
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(mpnn_ptr.parameters(), max_norm=1, norm_type=2)
            optim.step()
        print('Epoch: {}/{}, Loss: {}'.format(epoch+1, num_epochs, reward.mean().item()))
