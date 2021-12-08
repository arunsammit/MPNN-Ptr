#%%
from utils.datagenerate import generate_distance_matrix
import math
import torch
from torch_geometric.loader import DataLoader
from models.mpnn_ptr import MpnnPtr
from utils.utils import communication_cost, init_weights
from torch import nn
import matplotlib.pyplot as plt
import sys
from torch_geometric.data import Data
from scipy.stats import ttest_rel
#%%
# implement active search using a greedy rollout baseline instead of exponentially moving average baseline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = torch.load('./data/data_single_instance_42.pt', map_location=device)
graph_size = data.num_nodes
batch_size = 256
num_perm = 1024
datalist = []
#%%
for i in range(num_perm):
    perm = torch.randperm(graph_size,device=device)
    # permute nodes of the graph
    x = data.x[perm,:][:,perm]
    # permute edges of the graph
    edge_index = perm[data.edge_index]
    # create a Data object
    data_i = Data(x=x, edge_index=edge_index, edge_attr=data.edge_attr)
    datalist.append(data_i)
#%%
dataloader = DataLoader(datalist, batch_size = batch_size, shuffle=True)
dataloader_one_batch = DataLoader(datalist, batch_size = num_perm, shuffle=True)
# to be used for evaluation, maybe later replace it with another random permutation of data
data_batched = next(iter(dataloader_one_batch))
n = math.ceil(math.sqrt(graph_size))
m = math.ceil(graph_size / n)
distance_matrix = generate_distance_matrix(n,m).to(device)
mpnn_ptr = MpnnPtr(input_dim=graph_size, embedding_dim=graph_size + 10, hidden_dim=graph_size + 20, K=3, n_layers=2, p_dropout=0.1, device=device, logit_clipping=True)
mpnn_ptr_bst = MpnnPtr(input_dim=graph_size, embedding_dim=graph_size + 10, hidden_dim=graph_size + 20, K=3, n_layers=2, p_dropout=0.1, device=device, logit_clipping=True)
n = len(sys.argv)
if(n>1):
    mpnn_ptr.load_state_dict(torch.load(sys.argv[1]))
else:
    mpnn_ptr.apply(init_weights)
mpnn_ptr.to(device)
mpnn_ptr.train()
# copy mpnn_ptr to mpnn_ptr_bst
mpnn_ptr_bst.load_state_dict(mpnn_ptr.state_dict())
mpnn_ptr_bst.to(device)
mpnn_ptr_bst.eval()
optim = torch.optim.Adam(mpnn_ptr.parameters(), lr=0.0001)
best_cost = float('inf')
baseline = torch.tensor(0.0)
# data = next(iter(dataloader))
loss_list = []
num_epochs = 500
steps = 100
for epoch in range(num_epochs):
    for step in range(steps):
        best_cost_epoch = float('inf')
        for data in dataloader:
            num_samples = 1
            predicted_mappings, log_likelihood_sum = mpnn_ptr(data,num_samples)
            predicted_mappings.detach_()
            penalty = communication_cost(data.edge_index, data.edge_attr, data.batch, data.num_graphs, distance_matrix,
                                            predicted_mappings)
            arg_min_penalty = torch.argmin(penalty)
            best_cost_epoch = min(best_cost_epoch, penalty[arg_min_penalty].item())
            with torch.no_grad():
                baselines, _ = mpnn_ptr_bst(data,num_samples)
            loss = torch.mean((penalty.detach() - baseline.detach()) * log_likelihood_sum)
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(mpnn_ptr.parameters(), max_norm=1, norm_type=2)
            optim.step()
        print(f'Epoch: {epoch*steps + step}/{num_epochs*steps}, Loss: {best_cost_epoch}')
        loss_list.append(best_cost_epoch)
        if(best_cost_epoch < best_cost):
            best_cost = best_cost_epoch
    # perform one sided paired t-test on data_batched to determine if the current model is better than the baseline model
    predicted_mappings_current, _ = mpnn_ptr(data_batched,num_samples)
    penalty_current = communication_cost(data_batched.edge_index, data_batched.edge_attr, data_batched.batch, data_batched.num_graphs, distance_matrix, predicted_mappings_current)
    predicted_mappings_baseline, _ = mpnn_ptr_bst(data_batched,num_samples)
    penalty_baseline = communication_cost(data_batched.edge_index, data_batched.edge_attr, data_batched.batch, data_batched.num_graphs, distance_matrix, predicted_mappings_baseline)
    t_stat, p_value = ttest_rel(penalty_current.detach().cpu().numpy(), penalty_baseline.detach().cpu().numpy(),alternative='greater')
    if(p_value < 0.05):
        mpnn_ptr_bst.load_state_dict(mpnn_ptr.state_dict())

torch.save(mpnn_ptr.state_dict(), './models_data/model_single_42_v2.pt')
print('Best cost: {}'.format(best_cost))
# plot loss vs epoch
fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(loss_list)  # Plot some data on the axes.
ax.set_xlabel('number of epochs')  # Add an x-label to the axes.
ax.set_ylabel('communication cost')  # Add a y-label to the axes.
ax.set_title("communication cost v/s number of epochs")  # Add a title to the axes
fig.savefig('./plots/loss_single_42_v2.png')  # Save the figure.


