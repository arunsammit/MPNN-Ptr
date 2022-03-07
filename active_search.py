
from utils.datagenerate import generate_distance_matrix, generate_distance_matrix_3D
import math
import torch
from torch_geometric.loader import DataLoader
from models.mpnn_ptr import MpnnPtr
from utils.utils import  communication_cost_multiple_samples
from torch import nn
import matplotlib.pyplot as plt
import sys
from timeit import default_timer as timer
import argparse
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
data = torch.load(args.dataset, map_location=device)
graph_size = data.num_nodes
num_samples = args.num_samples
datalist = [data]
dataloader = DataLoader(datalist, batch_size=1)
if args.three_D:
    n = math.ceil(math.sqrt(graph_size/2))
    m = math.ceil(graph_size/ (n * 2))
    l = 2
    distance_matrix = generate_distance_matrix_3D(n, m, l).to(device)
else:
    n = math.ceil(math.sqrt(graph_size))
    m = math.ceil(graph_size / n)
    distance_matrix = generate_distance_matrix(n,m).to(device)
mpnn_ptr = MpnnPtr(input_dim=graph_size, embedding_dim=graph_size + 10, hidden_dim=graph_size + 20, K=3, n_layers=2, p_dropout=0.1, device=device, logit_clipping=True, feature_scale=290, decoding_type='sampling')
if args.pretrained_model_path is not None:
    mpnn_ptr.load_state_dict(torch.load(args.pretrained_model_path, map_location=device))
mpnn_ptr.to(device)
mpnn_ptr.train()
optim = torch.optim.Adam(mpnn_ptr.parameters(), lr=args.lr)
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=100, gamma=0.9)
best_mapping = None
best_cost = float('inf')
baseline = torch.tensor(0.0)
data = next(iter(dataloader))
loss_list = []
num_epochs = args.max_iter
count_not_decrease = 0
# start measuring time
start = timer()
for epoch in range(num_epochs):
    predicted_mappings, log_likelihood_sum = mpnn_ptr(data, num_samples)
    penalty = communication_cost_multiple_samples(data.edge_index, 
        data.edge_attr, data.batch, distance_matrix, predicted_mappings, num_samples)
    min_penalty = torch.argmin(penalty)
    if penalty[min_penalty] < best_cost:
        best_cost = penalty[min_penalty]
        best_mapping = predicted_mappings[min_penalty]
    if epoch == 0:
        baseline = penalty.mean()
    else:
        baseline = 0.9 * baseline + 0.1 * penalty.mean()
    loss = torch.mean((penalty.detach() - baseline.detach())*log_likelihood_sum)
    optim.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(mpnn_ptr.parameters(), max_norm=1, norm_type=2)
    optim.step()
    if epoch % 10 == 0:
        print(f'Epoch: {epoch + 1:4}/{num_epochs} Min Comm Cost: {penalty[min_penalty]:8.2f}   Avg Comm Cost: {penalty.mean():8.2f}')
    # break the training loop if min_penalty is not decreasing for consecutive 10000 epochs
    if penalty[min_penalty] > best_cost:
        count_not_decrease += 1
    else:
        count_not_decrease = 0
    if count_not_decrease > 20000:
        print('Early stopping at epoch {}'.format(epoch))
        break
    loss_list.append(penalty[min_penalty].item())
    # lr_scheduler.step()
# stop measuring time
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
# python3 active_search.py data_tgff/single/traffic_32.pt --lr 0.0001 --max_iter 5000 --num_samples 2048 --three_D