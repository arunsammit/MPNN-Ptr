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
from utils.utils import paired_t_test

#%%
# implement active search using a greedy rollout baseline instead of exponentially moving average baseline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = torch.load('./data/data_single_instance_uniform_64.pt', map_location=device)
graph_size = data.num_nodes
batch_size = 128
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
mpnn_ptr_baseline = MpnnPtr(input_dim=graph_size, embedding_dim=graph_size + 10, hidden_dim=graph_size + 20, K=3, n_layers=2, p_dropout=0.1, device=device, logit_clipping=True,decoding_type='greedy')
n = len(sys.argv)
if(n>1):
    mpnn_ptr.load_state_dict(torch.load(sys.argv[1]))
else:
    mpnn_ptr.apply(init_weights)
mpnn_ptr.to(device)
mpnn_ptr.train()
# copy mpnn_ptr to mpnn_ptr_bst
mpnn_ptr_baseline.load_state_dict(mpnn_ptr.state_dict())
mpnn_ptr_baseline.to(device)
mpnn_ptr_baseline.eval()
optim = torch.optim.Adam(mpnn_ptr.parameters(), lr=0.00001)
best_cost = float('inf')
baseline = torch.tensor(0.0)
# data = next(iter(dataloader))
num_epochs = 5000
loss_list = []
steps = 100
epoch_penalty = torch.zeros(len(dataloader))
num_repeats = math.ceil(num_epochs / steps)
@torch.no_grad()
def baseline_model_update(data_batched, distance_matrix, mpnn_ptr, mpnn_ptr_baseline):
    mpnn_ptr.eval()
    mpnn_ptr.decoding_type = 'greedy'
    predicted_mappings_current, _ = \
            mpnn_ptr(data_batched,1)
    mpnn_ptr.decoding_type ='sampling'
    penalty_current = \
            communication_cost(data_batched.edge_index, data_batched.edge_attr, data_batched.batch, data_batched.num_graphs, distance_matrix, predicted_mappings_current)
    predicted_mappings_baseline, _ = \
            mpnn_ptr_baseline(data_batched,1)
    penalty_baseline = \
            communication_cost(data_batched.edge_index, data_batched.edge_attr, data_batched.batch, data_batched.num_graphs, distance_matrix, predicted_mappings_baseline)
    p_value, t_value, mean = paired_t_test(penalty_current, penalty_baseline)
    print(f'p-value: {p_value}, t-value: {t_value}, mean: {mean}')
    if(p_value < 0.05):
        mpnn_ptr_baseline.load_state_dict(mpnn_ptr.state_dict())

for rep in range(num_repeats):
    for step in range(steps):
        best_cost_epoch = float('inf')
        epoch_penalty[:] = 0
        for i, data in enumerate(dataloader):
            num_samples = 1
            mpnn_ptr.train()
            predicted_mappings, log_likelihood_sum = mpnn_ptr(data,num_samples)
            predicted_mappings.detach_()
            penalty = \
                communication_cost(data.edge_index, data.edge_attr, data.batch, data.num_graphs, distance_matrix,predicted_mappings)
            epoch_penalty[i] = penalty.mean()
            arg_min_penalty = torch.argmin(penalty)
            best_cost_epoch = min(best_cost_epoch, penalty[arg_min_penalty].item())
            with torch.no_grad():
                mpnn_ptr_baseline.eval()
                baselines, _ = mpnn_ptr_baseline(data,1)
            loss = torch.mean((penalty.detach() - baseline.detach()) * log_likelihood_sum)
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(mpnn_ptr.parameters(), max_norm=1, norm_type=2)
            optim.step()
        num_epoch = rep * steps + step
        if(num_epoch % 10 == 0):
            print(f'Epoch: {num_epoch}/{num_epochs}, Loss: {epoch_penalty.mean().item():.4f}, Best Cost: {best_cost_epoch:.4f}')
        loss_list.append(best_cost_epoch)
        if(best_cost_epoch < best_cost):
            best_cost = best_cost_epoch
    baseline_model_update(data_batched, distance_matrix, mpnn_ptr, mpnn_ptr_baseline)

torch.save(mpnn_ptr.state_dict(), './models_data/model_single_64_uniform_v2.pt')
print('Best cost: {}'.format(best_cost))
# plot loss vs epoch
fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(loss_list)  # Plot some data on the axes.
ax.set_xlabel('number of epochs')  # Add an x-label to the axes.
ax.set_ylabel('communication cost')  # Add a y-label to the axes.
ax.set_title("communication cost v/s number of epochs")  # Add a title to the axes
fig.savefig('./plots/loss_single_64_uniform_v2.png')  # Save the figure.


