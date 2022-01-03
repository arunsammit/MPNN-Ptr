#%%
from torch_geometric.loader.dataloader import DataLoader
from models.mpnn_ptr import MpnnPtr
from utils.utils import communication_cost, init_weights
from utils.datagenerate import generate_distance_matrix
import torch
from torch import nn
import matplotlib.pyplot as plt
import math

#%%
# load data and generate distance matrix
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_path = 'data/data_single_64_2000.pt'
datalist = torch.load(data_path, map_location=device)
graph_size = datalist[0].num_nodes
n = math.ceil(math.sqrt(graph_size))
m = math.ceil(graph_size/n)
distance_matrix = generate_distance_matrix(n,m).to(device)
#%%
# create DataLoader
batch_size = 128
dataloader = DataLoader(datalist, batch_size, shuffle=True)
#%%
# initialize the models
mpnn_ptr = MpnnPtr(input_dim=graph_size, embedding_dim=graph_size + 10, hidden_dim=graph_size + 20, K=3, n_layers=2, p_dropout=0.1, device=device, logit_clipping=True)
mpnn_ptr.to(device)
mpnn_ptr.apply(init_weights)
# mpnn_ptr.load_state_dict(torch.load('models_data/model_pretrain_single_64_2.pt',map_location=device))
#%%
optim = torch.optim.Adam(mpnn_ptr.parameters(), lr=0.0001)
# learning rate schedular
lr_schedular = torch.optim.lr_scheduler.StepLR(optim, step_size=5000, gamma=0.96)
num_epochs = 200
penalty_baseline = None
epoch_penalty = torch.zeros(len(dataloader))
loss_list_pre = []
# loss_list_pre = torch.load('plots/loss_list_pre_2.pt')
#%%
for epoch in range(num_epochs):
    epoch_penalty[:] = 0
    for i, data in enumerate(dataloader):
        mpnn_ptr.train()
        predicted_mappings, log_likelihoods_sum = mpnn_ptr(data, 1)
        # predicted_mappings shape: (batch_size, max_graph_size_in_batch)
        # log_likelihoods_sum shape: (batch_size,)
        penalty = communication_cost(data.edge_index, data.edge_attr, data.batch, data.num_graphs, distance_matrix, predicted_mappings)
        epoch_penalty[i] = penalty.mean()
        if penalty_baseline is None:
            penalty_baseline = penalty.mean()
        else:
            penalty_baseline = 0.9 * penalty_baseline + 0.1 * penalty.mean()
        # penalty_baseline = calculate_baseline(data.edge_index, data.edge_attr, data.batch, data.num_graphs,  distance_matrix, samples)
        loss = torch.mean((penalty.detach() - penalty_baseline.detach()) * log_likelihoods_sum)
        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(mpnn_ptr.parameters(), max_norm=1, norm_type=2)
        optim.step()
        lr_schedular.step()
    batch_loss = epoch_penalty.mean().item()
    loss_list_pre.append(batch_loss)
    if epoch % 10 == 0:
        print('Epoch: {}/{}, Loss: {}'.format(epoch + 1, num_epochs, batch_loss))
#%%
# save the model
torch.save(mpnn_ptr.state_dict(), 'models_data/model_pretrain_single_64_2.pt')

#%%
# plot loss_list_pre
fig, ax = plt.subplots()
ax.plot(loss_list_pre)
ax.set_xlabel('Epoch')
ax.set_ylabel('Communication cost')
fig.savefig(f'plots/loss_list_pre_ema_{"".join(data_path.split("_")[1:3])}.png', dpi=300)

#%% save the loss list
torch.save(loss_list_pre, 'plots/loss_list_pre_2.pt')
# %%
