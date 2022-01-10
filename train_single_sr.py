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
from datetime import datetime
import argparse
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("datapath", help="path to dataset")
parser.add_argument("--epochs", help="number of epochs", type=int, default=100)
parser.add_argument("--num_samples", help="number of samples", type=int, default=8)
parser.add_argument("--batch_size", help="batch size", type=int, default=128)
parser.add_argument("--lr", help="learning rate", type=float, default=0.001)
parser.add_argument("--model_path", help="path to pretrained model", type=str, default=None)
parser.add_argument("--loss_list_path", help="path to loss list", type=str, default=None)
parser.add_argument("--lr_decay_rate", help="learning rate decay rate", type=float, default=0.9)
args = parser.parse_args()
#%%
# load data and generate distance matrix
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
datalist = torch.load(args.datapath, map_location=device)
graph_size = datalist[0].num_nodes
n = math.ceil(math.sqrt(graph_size))
m = math.ceil(graph_size/n)
dataloader = DataLoader(datalist, args.batch_size, shuffle=True)
distance_matrix = generate_distance_matrix(n,m).to(device)
#%%
# initialize the models
max_weight = 0
for i in range(len(datalist)):
    max_weight = max(max_weight, datalist[i].edge_attr.max())
mpnn_ptr = MpnnPtr(input_dim=graph_size, embedding_dim=graph_size + 10, hidden_dim=graph_size + 20, K=3, n_layers=2, p_dropout=0.1, device=device, logit_clipping=True, decoding_type='sampling', feature_scale=max_weight)
mpnn_ptr.to(device)
if args.model_path is not None:
    mpnn_ptr.load_state_dict(torch.load(args.model_path, map_location=device))
else:
    mpnn_ptr.apply(init_weights)
#%%
# initializing the optimizer
optim = torch.optim.Adam(mpnn_ptr.parameters(), lr=args.lr)
lr_schedular = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.9)
penalty_baseline = None
epoch_penalty = torch.zeros(len(dataloader))
loss_list_pre = torch.load(args.loss_list_path) if args.loss_list_path else []
mpnn_ptr.train()
#%%
for epoch in range(args.epochs):
    for i, data in enumerate(dataloader):
        predicted_mappings, log_likelihoods_sum = mpnn_ptr(data, args.num_samples)
        penalty, baseline = communication_cost_multiple_samples(data.edge_index, data.edge_attr, data.batch, distance_matrix, predicted_mappings, args.num_samples, calculate_baseline=True)
        epoch_penalty[i] = penalty.mean()
        penalty_baseline = baseline.repeat(args.num_samples)
        loss = torch.mean((penalty.detach() - penalty_baseline.detach()) * log_likelihoods_sum)
        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(mpnn_ptr.parameters(), max_norm=1, norm_type=2)
        optim.step()
    lr_schedular.step()
    batch_loss = epoch_penalty.mean().item()
    loss_list_pre.append(batch_loss)
    print('Epoch: {}/{}, Loss: {}'.format(epoch + 1, args.epochs, batch_loss))
#%%
# save the model
datetime_suffix = datetime.now().strftime('%m-%d_%H-%M')
Path('models_data').mkdir(parents=True, exist_ok=True)
torch.save(mpnn_ptr.state_dict(), f'models_data/model_pretrain_sr_{graph_size}_{datetime_suffix}.pt')

#%%
# plot loss_list_pre
fig, ax = plt.subplots()
ax.plot(loss_list_pre)
ax.set_xlabel('Epoch')
ax.set_ylabel('Communication cost')
# save figure
Path('plots').mkdir(parents=True, exist_ok=True)
fig.savefig(f'plots/loss_list_pre_sr_{graph_size}_{datetime_suffix}.png', dpi=300)

#%% save the loss list
torch.save(loss_list_pre, f'plots/loss_list_pre_sr_{graph_size}_{datetime_suffix}.pt')
# %%
