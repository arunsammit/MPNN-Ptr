#%%
from torch_geometric.loader.dataloader import DataLoader
from models.mpnn_ptr import MpnnPtr
from utils.baseline import calculate_baseline
from utils.utils import communication_cost_multiple_samples , init_weights
from utils.datagenerate import generate_distance_matrix
import torch
from torch import nn
import matplotlib.pyplot as plt
import math
import sys
from datetime import datetime

#%%
if len(sys.argv) < 3:
    print('Usage: python3 train_single_simple.py <dataset> <max_iter> <batch_size (default =128)> <num_samples (default=4)> <pretrained_model_path (optional)> <loss_list_path (optional)>')
    sys.exit()

#%%
# load data and generate distance matrix
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
datapath = sys.argv[1]
# datalist = torch.load('data/data_single_64_2000.pt',map_location=device)
datalist = torch.load(datapath, map_location=device)
graph_size = datalist[0].num_nodes
n = math.ceil(math.sqrt(graph_size))
m = math.ceil(graph_size/n)
distance_matrix = generate_distance_matrix(n,m).to(device)
#%%
# create DataLoader
batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 128
dataloader = DataLoader(datalist, batch_size, shuffle=True)
#%%
# initialize the models
mpnn_ptr = MpnnPtr(input_dim=graph_size, embedding_dim=graph_size + 10, hidden_dim=graph_size + 20, K=3, n_layers=2, p_dropout=0.1, device=device, logit_clipping=True)
mpnn_ptr.to(device)
if len(sys.argv) > 5:
    mpnn_ptr.load_state_dict(torch.load(sys.argv[5],map_location=device))
else:
    mpnn_ptr.apply(init_weights)
# mpnn_ptr.load_state_dict(torch.load('models_data/model_pretrain_single_64_2.pt',map_location=device))
#%%
optim = torch.optim.Adam(mpnn_ptr.parameters(), lr=0.0001)
# learning rate schedular
# lr_schedular = torch.optim.lr_scheduler.StepLR(optim, step_size=5000, gamma=0.96)
num_epochs = int(sys.argv[2])
penalty_baseline = None
epoch_penalty = torch.zeros(len(dataloader))
loss_list_pre = torch.load(sys.argv[6]) if len(sys.argv) > 6 else torch.zeros(num_epochs)
num_samples = int(sys.argv[4]) if len(sys.argv) > 4 else 4
mpnn_ptr.train()
#%%
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader):
        predicted_mappings, log_likelihoods_sum = mpnn_ptr(data, num_samples)
        penalty, baseline = communication_cost_multiple_samples(data.edge_index, data.edge_attr, data.batch, distance_matrix, predicted_mappings, num_samples, calculate_baseline=True)
        epoch_penalty[i] = penalty.mean()
        penalty_baseline = baseline.repeat(num_samples)
        loss = torch.mean((penalty.detach() - penalty_baseline.detach()) * log_likelihoods_sum)
        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(mpnn_ptr.parameters(), max_norm=1, norm_type=2)
        optim.step()
        # lr_schedular.step()
    batch_loss = epoch_penalty.mean().item()
    loss_list_pre.append(batch_loss)
    print('Epoch: {}/{}, Loss: {}'.format(epoch + 1, num_epochs, batch_loss))
#%%
# save the model
datetime_suffix = datetime.now().strftime('%m-%d_%H-%M')
torch.save(mpnn_ptr.state_dict(), f'models_data/model_pretrain_sr_{graph_size}_{datetime_suffix}.pt')

#%%
# plot loss_list_pre
fig, ax = plt.subplots()
ax.plot(loss_list_pre)
ax.set_xlabel('Epoch')
ax.set_ylabel('Communication cost')
# save figure

fig.savefig(f'plots/loss_list_pre_sr_{graph_size}_{datetime_suffix}.png', dpi=300)

#%% save the loss list
torch.save(loss_list_pre, f'plots/loss_list_pre_sr_{graph_size}_{datetime_suffix}.pt')
# %%
