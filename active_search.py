
from utils.datagenerate import generate_distance_matrix
import math
import torch
from torch_geometric.loader import DataLoader
from models.mpnn_ptr import MpnnPtr
from utils.utils import communication_cost
from torch import nn
import matplotlib.pyplot as plt
import sys
#%%

args_len = len(sys.argv)
if args_len < 3:
    print('Usage: python3 active_search.py <dataset> <max_iter> <batch_size (default =128)> <pretrained_model_path (optional)> ')
    sys.exit()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = torch.load(sys.argv[1])
graph_size = data.num_nodes
if args_len>3:
    batch_size = int(sys.argv[3])
else:
    batch_size = 128
datalist = [data for _ in range(batch_size)]
dataloader = DataLoader(datalist, batch_size=batch_size)
n = math.ceil(math.sqrt(graph_size))
m = math.ceil(graph_size / n)
distance_matrix = generate_distance_matrix(n,m).to(device)
mpnn_ptr = MpnnPtr(input_dim=graph_size, embedding_dim=graph_size + 10, hidden_dim=graph_size + 20, K=3, n_layers=2, p_dropout=0.1, device=device, logit_clipping=True)
if(args_len>4):
    mpnn_ptr.load_state_dict(torch.load(sys.argv[4]))
mpnn_ptr.to(device)
mpnn_ptr.train()
optim = torch.optim.Adam(mpnn_ptr.parameters(), lr=0.0001)
best_mapping = None
best_cost = float('inf')
baseline = torch.tensor(0.0)
data = next(iter(dataloader))

loss_list = []
num_epochs = int(sys.argv[2])
count_not_decrease = 0
for epoch in range(num_epochs):
    num_samples = 1
    predicted_mappings, log_likelihood_sum = mpnn_ptr(data,num_samples)
    predicted_mappings.detach_()
    penalty = communication_cost(data.edge_index, data.edge_attr, data.batch, data.num_graphs, distance_matrix,
                                    predicted_mappings)
    min_penalty = torch.argmin(penalty)
    if penalty[min_penalty] < best_cost:
        best_cost = penalty[min_penalty]
        best_mapping = predicted_mappings[min_penalty]
    if epoch == 0:
        baseline = penalty.mean()
    else:
        baseline = 0.99 * baseline + 0.01 * penalty.mean()
    loss = torch.mean((penalty.detach() - baseline.detach())*log_likelihood_sum)
    optim.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(mpnn_ptr.parameters(), max_norm=1, norm_type=2)
    optim.step()
    if epoch % 10 == 0:
        print('Epoch: {}/{}, Loss: {} '.format(epoch + 1, num_epochs, penalty.mean()))
    # break the training loop if min_penalty is not decreasing for consecutive 10000 epochs
    if penalty[min_penalty] > best_cost:
        count_not_decrease += 1
    else:
        count_not_decrease = 0
    if count_not_decrease > 20000:
        break    
    loss_list.append(penalty.mean().item())
    # lr_scheduler.step()
torch.save(mpnn_ptr.state_dict(), f'./models_data/model_single_uniform_{graph_size}.pt')
print('Best cost: {}'.format(best_cost))
# plot loss vs epoch
fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(loss_list)  # Plot some data on the axes.
ax.set_xlabel('number of epochs')  # Add an x-label to the axes.
ax.set_ylabel('communication cost')  # Add a y-label to the axes.
ax.set_title("communication cost v/s number of epochs")  # Add a title to the axes
fig.savefig(f'./plots/loss_single_uniform_{graph_size}_2.png')  # Save the figure.


