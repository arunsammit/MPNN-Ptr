#%%
from torch_geometric.loader.dataloader import DataLoader
from torch_geometric.data import Data
from models.mpnn_ptr import MpnnPtr
from utils.utils import init_weights
from utils.datagenerate import generate_distance_matrix, DistanceMatrix
import torch
from torch import nn
import matplotlib.pyplot as plt
import math
import sys
from datetime import datetime
from graphdataset import MultipleGraphDataset, getDataLoader
from train.trainers import TrainerInitPop, TrainerSR
from train.validation import validate_dataloader
#%% initializing the parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_graph_size = 49
batch_size_train = 128
batch_size_dev = 128
saved_model_path = None
lr = 0.001
lr_decay_gamma = .96
num_epochs = 100
num_samples = 8
training_algorithm = 'pretrain' # 'init_pop' or 'pretrain'
#%%
root_train = 'data_tgff/multiple/train'
root_dev = 'data_tgff/multiple/test'
train_good_files = ['traindata_multiple_TGFF_norm_12.pt', 'traindata_multiple_TGFF_norm_16.pt', 'traindata_multiple_TGFF_norm_20.pt', 'traindata_multiple_TGFF_norm_36.pt', 'traindata_multiple_TGFF_norm_49.pt']
dev_good_files = ['testdata_multiple_TGFF_norm_16.pt', 'testdata_multiple_TGFF_norm_32.pt']
train_dataloader = getDataLoader('data_tgff/multiple/train', batch_size_train, max_graph_size = max_graph_size, raw_file_names = train_good_files)
dev_dataloader = getDataLoader('data_tgff/multiple/test', batch_size_dev, max_graph_size = max_graph_size, raw_file_names = dev_good_files)
#%% initialize the models
mpnn_ptr = MpnnPtr(input_dim=max_graph_size, embedding_dim=max_graph_size + 10, hidden_dim=max_graph_size + 20, K=3, n_layers=2, p_dropout=0.1, device=device, logit_clipping=True)
mpnn_ptr.to(device)
#%% load model if saved
if saved_model_path:
    mpnn_ptr.load_state_dict(torch.load(saved_model_path,map_location=device))
else:
    mpnn_ptr.apply(init_weights)
#%% initialize the training algorithm
if training_algorithm == 'init_pop':
    trainer = TrainerInitPop(mpnn_ptr, num_samples + 1)
elif training_algorithm == 'pretrain':
    trainer = TrainerSR(mpnn_ptr, num_samples)
distance_matrix_dict = DistanceMatrix()
optim = torch.optim.Adam(mpnn_ptr.parameters(), lr=lr)
lr_schedular = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=lr_decay_gamma)
loss_list_train = []
loss_list_dev = []
#%% training starts
for epoch in range(num_epochs):
    avg_train_comm_cost = trainer.train(train_dataloader, distance_matrix_dict, optim)
    lr_schedular.step()
    loss_list_train.append(avg_train_comm_cost)
    avg_valid_comm_cost = validate_dataloader(mpnn_ptr,dev_dataloader, distance_matrix_dict, 100)
    loss_list_dev.append(avg_valid_comm_cost)
    print(f'Epoch: {epoch + 1}/{num_epochs}, Train Comm Cost: {avg_train_comm_cost:.4f}, Dev Comm Cost: {avg_valid_comm_cost:.4f}')
#%%
# save the model
datetime_suffix = datetime.now().strftime('%m-%d_%H-%M')
torch.save(mpnn_ptr.state_dict(), f'models_data/model_{training_algorithm}_{max_graph_size}_{datetime_suffix}.pt')

#%%
# plot loss_list_pre
fig, ax = plt.subplots()
ax.plot(loss_list_train)
ax.plot(loss_list_dev)
ax.set_xlabel('Epoch')
ax.set_ylabel('Communication cost')
fig.savefig(f'plots/loss_list_{max_graph_size}_{datetime_suffix}.png', dpi=300)

#%% save the loss list
torch.save(loss_list_train, f'plots/loss_list_train_{training_algorithm}_{max_graph_size}_{datetime_suffix}.pt')
torch.save(loss_list_dev, f'plots/loss_list_dev_{training_algorithm}_{max_graph_size}_{datetime_suffix}.pt')
# %%
