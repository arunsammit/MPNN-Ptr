if __name__ == '__main__':
    from models.mpnn_ptr import MpnnPtr
    from utils.utils import communication_cost, calculate_baseline
    import torch
    from torch import nn
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader, distance_matrices = torch.load('data/data_single_49.pt')
    max_graph_size = 49
    mpnn_ptr = MpnnPtr(input_dim=max_graph_size, embedding_dim=55, hidden_dim=60, K=3, n_layers=2,
                       p_dropout=0.1,device=device)
    mpnn_ptr.to(device)
    optim = torch.optim.Adam(mpnn_ptr.parameters(), lr=0.0001)
    num_epochs = 100
    for epoch in range(num_epochs):
        for data, distance_matrix in zip(dataloader, distance_matrices):
            num_samples = 5
            mpnn_ptr.train()
            samples, predicted_mappings, log_likelihoods_sum = mpnn_ptr(data,num_samples)
            # samples shape: (batch_size, num_samples, max_graph_size_in_batch)
            # predicted_mappings shape: (batch_size, max_graph_size_in_batch)
            # log_likelihoods_sum shape: (batch_size,)
            penalty = communication_cost(data.edge_index, data.edge_attr, data.batch, data.num_graphs, distance_matrix, predicted_mappings)
            penalty_baseline = calculate_baseline(data.edge_index, data.edge_attr, data.batch, data.num_graphs, distance_matrix, samples)
            loss = (1/data.num_graphs) * torch.sum((penalty.detach() - penalty_baseline.detach()) * log_likelihoods_sum)
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(mpnn_ptr.parameters(), max_norm=1, norm_type=2)
            optim.step()
        print('Epoch: {}/{}, Loss: {}'.format(epoch + 1, num_epochs, penalty.mean().item()))
