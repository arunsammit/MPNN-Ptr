#%%
from models.mpnn_ptr import MpnnPtr
from utils.utils import communication_cost, calculate_baseline, init_weights
import torch
from torch import nn
import matplotlib.pyplot as plt

#%%

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataloader, distance_matrices = torch.load('data/data_single_64.pt',map_location=device)
max_graph_size = 64
mpnn_ptr = MpnnPtr(input_dim=max_graph_size, embedding_dim=75, hidden_dim=81, K=3, n_layers=4, p_dropout=0.1, logit_clipping=True, device=device)
mpnn_ptr.to(device)
mpnn_ptr.apply(init_weights)
optim = torch.optim.Adam(mpnn_ptr.parameters(), lr=0.00001)
num_epochs = 10000
epoch_penalty = torch.zeros(len(dataloader))
loss_list_pre = []
#%%
for epoch in range(num_epochs):
    epoch_penalty[:] = 0
    for i, (data, distance_matrix) in enumerate(zip(dataloader, distance_matrices)):
        num_samples = 32
        mpnn_ptr.train()
        samples, predicted_mappings, log_likelihoods_sum = mpnn_ptr(data, num_samples)
        # samples shape: (batch_size, num_samples, max_graph_size_in_batch)
        # predicted_mappings shape: (batch_size, max_graph_size_in_batch)
        # log_likelihoods_sum shape: (batch_size,)
        penalty = communication_cost(data.edge_index, data.edge_attr, data.batch, data.num_graphs, distance_matrix, predicted_mappings)
        epoch_penalty[i] = penalty.mean()
        penalty_baseline = calculate_baseline(data.edge_index, data.edge_attr, data.batch, data.num_graphs,  distance_matrix, samples)
        loss = torch.mean((penalty.detach() - penalty_baseline.detach()) * log_likelihoods_sum)
        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(mpnn_ptr.parameters(), max_norm=1, norm_type=2)
        optim.step()
    batch_loss = epoch_penalty.mean().item()
    loss_list_pre.append(batch_loss)
    print('Epoch: {}/{}, Loss: {}'.format(epoch + 1, num_epochs, batch_loss))
#%%
# save the model
torch.save(mpnn_ptr.state_dict(), 'models_data/model_pretrain_single_64.pt')
#%%
# save loss_list_pre
torch.save(loss_list_pre, 'models_data/loss_list_pretrain_single_64.pt')

#%%
# plot loss_list_pre
fig, ax = plt.subplots()
ax.plot(loss_list_pre)
ax.set_xlabel('Epoch')
ax.set_ylabel('Communication cost')
# save figure

fig.savefig('plots/loss_list_pre.png', dpi=300)

# %%
