from utils.datagenerate import generate_graph_data_list, generate_distance_matrix
import math
import torch
from torch_geometric.loader import DataLoader
from models.mpnn_ptr import MpnnPtr
from utils.utils import communication_cost
from torch import nn
if __name__ == "__main__":
    graph_size = 49
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = generate_graph_data_list(graph_size, graph_size, 1)[0].to(device)
    batch_size = 10
    datalist = [data for _ in range(batch_size)]
    dataloader = DataLoader(datalist, batch_size=batch_size)
    n = math.ceil(math.sqrt(graph_size))
    m = math.ceil(math.sqrt(graph_size))
    distance_matrix = generate_distance_matrix(7,7).to(device)
    mpnn_ptr = MpnnPtr(input_dim=graph_size, embedding_dim=55, hidden_dim=60, K=3, n_layers=2,
                       p_dropout=0.1, device=device)
    mpnn_ptr.to(device)
    mpnn_ptr.train()
    optim = torch.optim.Adam(mpnn_ptr.parameters(), lr=0.0001)
    num_epochs = 1000
    best_mapping = None
    best_cost = float('inf')
    baseline = torch.tensor(0.0)
    for epoch in range(num_epochs):
        for data in dataloader:
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
        print('Epoch: {}/{}, Loss: {} '.format(epoch + 1, num_epochs, best_cost))


