#%%
from typing import List, Union
from torch_geometric.loader.dataloader import DataLoader
from models.mpnn_ptr import MpnnPtr
from utils.utils import communication_cost_multiple_samples , init_weights
from utils.datagenerate import generate_distance_matrix
import torch
from torch import nn
import matplotlib.pyplot as plt
import math
from datetime import datetime
import argparse
from pathlib import Path
import time

#%%
# load data and generate distance matrix
#%%
# initialize the models
def initialize(model_path, datapath, batch_size, device):
    datalist = torch.load(datapath, map_location=device)
    graph_size = datalist[0].num_nodes
    dataloader = DataLoader(datalist, batch_size, shuffle=True)
    max_weight = 0
    for i in range(len(datalist)):
        max_weight = max(max_weight, datalist[i].edge_attr.max())
    mpnn_ptr = MpnnPtr(input_dim=graph_size, embedding_dim=graph_size + 10, hidden_dim=graph_size + 20, K=3, n_layers=2, p_dropout=0.1, device=device, logit_clipping=True, decoding_type='sampling', feature_scale=max_weight)
    mpnn_ptr.to(device)
    if model_path is not None:
        mpnn_ptr.load_state_dict(torch.load(model_path, map_location=device))
    else:
        mpnn_ptr.apply(init_weights)
    return mpnn_ptr, dataloader
#%%
# initializing the optimizer

def train_step(mpnn_ptr, dataloader, distance_matrix, optimizer, num_samples):
    mpnn_ptr.train()
    epoch_loss = 0.0
    for data in dataloader:
        optimizer.zero_grad()
        data = data.to(mpnn_ptr.device)
        mappings, ll_sum = mpnn_ptr(data, num_samples)
        penalty, baseline = communication_cost_multiple_samples(data.edge_index, data.edge_attr, data.batch, distance_matrix, mappings, num_samples, calculate_baseline=True)
        epoch_loss += float(penalty.sum())
        penalty_baseline = baseline.repeat(num_samples)
        loss = torch.mean((penalty.detach() - penalty_baseline.detach()) * ll_sum)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mpnn_ptr.parameters(), max_norm=1, norm_type=2)
        optimizer.step()
    return epoch_loss / len(dataloader.dataset)
#%%
def train(mpnn_ptr, dataloaders:Union[List[DataLoader],DataLoader], num_samples, epochs, lr, lr_decay_rate, lr_step_size = 10, loss_list = None):
    optimizer = torch.optim.Adam(mpnn_ptr.parameters(), lr=lr)
    lr_schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_decay_rate)
    if not loss_list:
        loss_list = []
    if not isinstance(dataloaders, List):
        dataloaders = [dataloaders]
    for epoch in range(epochs):
        epoch_loss = 0.0
        start_time = time.time()
        for dataloader in dataloaders:
            epoch_loss += train_step(mpnn_ptr, dataloader, optimizer, num_samples)
        end_time = time.time()
        lr_schedular.step()
        loss_list.append(epoch_loss)
        time_taken = end_time - start_time
        minutes = int(time_taken / 60)
        seconds = int(time_taken % 60)
        print(f'Epoch: {epoch + 1}/{epochs}, Loss: {epoch_loss}, Time: {minutes}m {seconds}s')
    return loss_list
#%%
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_list = torch.load(args.loss_list_path) if args.loss_list_path else []
    mpnn_ptr, dataloader = initialize(args.model_path, args.datapath, args.batch_size, device)
    loss_list = train(mpnn_ptr, dataloader, args.num_samples, args.epochs, args.lr, args.lr_decay_rate, loss_list=loss_list)
    datetime_suffix = datetime.now().strftime('%m-%d_%H-%M')
    graph_size = dataloader.dataset[0].num_nodes
    Path('models_data').mkdir(parents=True, exist_ok=True)
    torch.save(mpnn_ptr.state_dict(), f'models_data/model_pretrain_sr_{graph_size}_{datetime_suffix}.pt')
    fig, ax = plt.subplots()
    ax.plot(loss_list)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Communication cost')
    # save figure
    Path('plots').mkdir(parents=True, exist_ok=True)
    fig.savefig(f'plots/loss_list_pre_sr_{graph_size}_{datetime_suffix}.png', dpi=300)
    torch.save(loss_list, f'plots/loss_list_pre_sr_{graph_size}_{datetime_suffix}.pt')
# %%
if __name__ == '__main__':
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
    main(args)
