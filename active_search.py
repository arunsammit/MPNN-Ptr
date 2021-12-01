
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = torch.load('./data/data_single_instance_64.pt')
graph_size = data.num_nodes
batch_size = 128
datalist = [data for _ in range(batch_size)]
dataloader = DataLoader(datalist, batch_size=batch_size)
n = math.ceil(math.sqrt(graph_size))
m = math.ceil(graph_size / n)
distance_matrix = generate_distance_matrix(n,m).to(device)
mpnn_ptr = MpnnPtr(input_dim=graph_size, embedding_dim=graph_size + 10, hidden_dim=graph_size + 20, K=3, n_layers=2, p_dropout=0.1, device=device, logit_clipping=True)
n = len(sys.argv)
if(n>1):
    mpnn_ptr.load_state_dict(torch.load(sys.argv[1]))
mpnn_ptr.to(device)
mpnn_ptr.train()
optim = torch.optim.Adam(mpnn_ptr.parameters(), lr=0.0001)
best_mapping = None
best_cost = float('inf')
baseline = torch.tensor(0.0)
data = next(iter(dataloader))
loss_list = []
num_epochs = 10000
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
    print('Epoch: {}/{}, Loss: {} '.format(epoch + 1, num_epochs, penalty[min_penalty]))
    loss_list.append(penalty[min_penalty].item())
    # lr_scheduler.step()
torch.save(mpnn_ptr.state_dict(), './models_data/model_single_64.pt')
print('Best cost: {}'.format(best_cost))
# plot loss vs epoch
fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(loss_list)  # Plot some data on the axes.
ax.set_xlabel('number of epochs')  # Add an x-label to the axes.
ax.set_ylabel('communication cost')  # Add a y-label to the axes.
ax.set_title("communication cost v/s number of epochs")  # Add a title to the axes
fig.savefig('./plots/loss_single_64.png')  # Save the figure.


