def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

if __name__ == '__main__':
    from models.mpnn_ptr import MpnnPtr
    from utils.utils import communication_cost, calculate_baseline
    import torch
    from torch import nn
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader, distance_matrices = torch.load('data/data_single_64.pt')
    max_graph_size = 64
    mpnn_ptr = MpnnPtr(input_dim=max_graph_size, embedding_dim=75, hidden_dim=81, K=3, n_layers=4, p_dropout=0,
                       logit_clipping=True, device=device)
    mpnn_ptr.to(device)
    mpnn_ptr.apply(init_weights)
    optim = torch.optim.Adam(mpnn_ptr.parameters(), lr=0.0001)
    num_epochs = 3500
    epoch_penalty = torch.zeros(len(dataloader))
    loss_list_pre = []
    for epoch in range(num_epochs):
        epoch_penalty[:] = 0
        for i, (data, distance_matrix) in enumerate(zip(dataloader, distance_matrices)):
            num_samples = 16
            mpnn_ptr.train()
            samples, predicted_mappings, log_likelihoods_sum = mpnn_ptr(data, num_samples)
            # samples shape: (batch_size, num_samples, max_graph_size_in_batch)
            # predicted_mappings shape: (batch_size, max_graph_size_in_batch)
            # log_likelihoods_sum shape: (batch_size,)
            penalty = communication_cost(data.edge_index, data.edge_attr, data.batch, data.num_graphs, distance_matrix,
                                         predicted_mappings)
            epoch_penalty[i] = penalty.mean()
            penalty_baseline = calculate_baseline(data.edge_index, data.edge_attr, data.batch, data.num_graphs,
                                                  distance_matrix, samples)
            loss = torch.mean((penalty.detach() - penalty_baseline.detach()) * log_likelihoods_sum)
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(mpnn_ptr.parameters(), max_norm=1, norm_type=2)
            optim.step()
        batch_loss = epoch_penalty.mean().item()
        loss_list_pre.append(batch_loss)
        print('Epoch: {}/{}, Loss: {}'.format(epoch + 1, num_epochs, batch_loss))