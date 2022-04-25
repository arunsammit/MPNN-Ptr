#%%
from utils.datagenerate import generate_distance_matrix, generate_distance_matrix_3D
import math
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from models.mpnn_ptr import MpnnPtr
from utils.utils import  communication_cost_multiple_samples
from torch import nn
import matplotlib.pyplot as plt
from train.validation import beam_search_data
from timeit import default_timer as timer
import argparse
from gumbel import gumbel_log_survival
from active_search import load_and_process_data, load_model
#%%
parser = argparse.ArgumentParser()
parser.add_argument('dataset', help='path to dataset', type=str)
parser.add_argument('--max_iter', help='max iterations', type=int, default=10000)
parser.add_argument('--num_samples', help='number of unique solutions to be sampled in each iteration', type=int, default=128)
parser.add_argument('--pretrained_model_path', help='path to pretrained model', type=str, default=None)
parser.add_argument('--lr', help='learning rate', type=float, default=0.001)
parser.add_argument('--three_D', help='use a fully connected 3D NoC with 2 layers in the Z direction', action='store_true')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%
num_samples = args.num_samples
max_graph_size = 121
data = load_and_process_data(args.dataset, device)
graph_size = data.num_nodes
if args.three_D:
    n = math.ceil(math.sqrt(graph_size/2))
    m = math.ceil(graph_size/ (n * 2))
    l = 2
    distance_matrix = generate_distance_matrix_3D(n, m, l).to(device)
else:
    n = math.floor(math.sqrt(graph_size))
    m = math.ceil(graph_size / n)
    distance_matrix = generate_distance_matrix(n,m).to(device)
mpnn_ptr = load_model(max_graph_size, device, feature_scale=1, pretrained_model_path=args.pretrained_model_path)
mpnn_ptr.train()
optim = torch.optim.Adam(mpnn_ptr.parameters(), lr=args.lr)
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=100, gamma=0.93)
best_mapping = None
best_cost = float('inf')
loss_list = []
num_epochs = args.max_iter
count_not_decrease = 0
# start measuring time
start = timer()
for epoch in range(num_epochs):
    mpnn_ptr.train()
    mpnn_ptr.decoding_type = 'sampling-w/o-replacement'
    mappings, log_probs, g_log_probs = mpnn_ptr(data, num_samples + 1)
    mappings = mappings[:-1, :]
    threshold = g_log_probs[-1]
    log_probs = log_probs[:-1]
    g_log_probs = g_log_probs[:-1]
    penalty = communication_cost_multiple_samples(data.edge_index, 
        data.edge_attr, data.batch, distance_matrix, mappings, num_samples)
    # take the value of the last sample for threshold
    log_q = gumbel_log_survival(threshold - g_log_probs).detach()
    log_importance = (log_probs - log_q).detach()
    wi_s = (log_importance.unsqueeze(1).repeat(1,log_importance.size(0)) - log_importance.unsqueeze(0)).exp().sum(dim=0) - 1 + torch.exp(log_q)
    importance_normalized = F.softmax(log_importance, dim=0)
    b_s = torch.sum( importance_normalized * penalty)
    min_penalty = torch.argmin(penalty)
    if penalty[min_penalty] < best_cost:
        best_cost = penalty[min_penalty]
        best_mapping = mappings[min_penalty]
    loss = torch.sum((1/wi_s) * log_probs * (penalty - b_s))
    optim.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(mpnn_ptr.parameters(), max_norm=1, norm_type=2)
    optim.step()
    if epoch % 20 == 0:
        mpnn_ptr.eval()
        mpnn_ptr.decoding_type = 'greedy'
        mapping, cost = beam_search_data(mpnn_ptr, data, distance_matrix, 1024)
        if cost < best_cost:
            best_cost = float(cost)
            best_mapping = mapping[0]
        print(f'Epoch: {epoch + 1:4}/{num_epochs} Min Comm Cost: {best_cost:8.2f}   Avg Comm Cost: {b_s:8.2f}')
        print(f'{best_mapping}')

    # break the training loop if min_penalty is not decreasing for consecutive 2000 epochs
    if penalty[min_penalty] > best_cost:
        count_not_decrease += 1
    else:
        count_not_decrease = 0
    if count_not_decrease > 5000:
        print('Early stopping at epoch {}'.format(epoch))
        break
    loss_list.append(penalty[min_penalty].item())
    # lr_scheduler.step()
# stop measuring time
# use the model with the best cost to do greedy beam search
mpnn_ptr.eval()
mpnn_ptr.decoding_type = 'greedy'
mapping, cost = beam_search_data(mpnn_ptr, data, distance_matrix, 3072)
if cost < best_cost:
    best_cost = float(cost)
    best_mapping = mapping[0]
end = timer()
torch.save(mpnn_ptr.state_dict(), f'./models_data/model_single_uniform_{graph_size}.pt')
print(f'Best cost: {best_cost}, time taken: {end - start}')
print(f'Best mapping: {best_mapping}')
# plot loss vs epoch
fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(loss_list)  # Plot some data on the axes.
ax.set_xlabel('number of epochs')  # Add an x-label to the axes.
ax.set_ylabel('communication cost')  # Add a y-label to the axes.
ax.set_title("communication cost v/s number of epochs")  # Add a title to the axes
fig.savefig(f'./plots/loss_single_uniform_{graph_size}_3.png')  # Save the figure.

# command to run with pretrained model:
# python3 active_search.py data_tgff/single/traffic_32.pt --lr 0.002 --pretrained_model_path models_data_final/model_16_01-10.pt --max_iter 5000 --num_samples 2048 --three_D
# command to run without pretrained model:
# python3 active_search.py data_tgff/single/traffic_72.pt --lr 0.0002 --max_iter 10000 --num_samples 1024
# command to run with pretrained model and 3D:
# python3 active_search.py data_tgff/single/traffic_32.pt --lr 0.001 --max_iter 5000 --num_samples 1024 --three_D