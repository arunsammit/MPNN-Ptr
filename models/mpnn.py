import torch
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_geometric.loader import DataLoader
from torch_scatter import scatter


class ConnectionsEmbedding(MessagePassing):
    def __init__(self, in_channels, n_features):
        super(ConnectionsEmbedding, self).__init__(aggr='mean')
        self.layer2 = nn.Sequential(
            nn.Linear(in_channels + 1, n_features - 1, bias=False),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(n_features, n_features, bias=False),
            nn.ReLU()
        )

    def message(self, x_j, edge_attr):
        # print(x_j.shape, edge_attr.shape)
        return self.layer2(torch.cat([edge_attr, x_j], dim=1))

    def update(self, aggr_out: Tensor, edge_index: Tensor) -> Tensor:
        # Concatenate node degree to node features.
        # x: [N, input_node_dim]
        # edge_index: [2, E]
        # print(aggr_out.shape, edge_index.shape)
        row, col = edge_index
        deg = degree(col, aggr_out.size(0), dtype=aggr_out.dtype)
        return self.layer3(torch.cat([aggr_out, deg.unsqueeze(-1)], dim=1))

    def forward(self, x, edge_index, edge_attr):
        # x: [N, input_node_dim]
        # edge_index: [2, E]
        # edge_attr: [E, 1]
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)


class Mpnn(MessagePassing):
    def __init__(self, input_node_dim, n_features, K):
        # input_node_dim: dimension of input node features.
        # n_features: dimension of the output embeddings
        super().__init__(aggr='add')
        self.K = K
        self.node_init_embedding_layer = nn.Sequential(
            nn.Linear(input_node_dim, n_features, bias=False),
            nn.ReLU()
        )
        self.theta4layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * n_features, n_features, bias=False),
                nn.ReLU()
            )
            for _ in range(K)
        ]
        )
        self.theta5layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * n_features, n_features, bias=False),
                nn.ReLU()
            )
            for _ in range(K)
        ]
        )
        self.theta6 = nn.Sequential(
            nn.Linear(n_features, n_features, bias=False),
            nn.ReLU()
        )
        self.theta7 = nn.Linear(2 * n_features, n_features, bias=False)
        self.connection_embedding_layer = ConnectionsEmbedding(in_channels=input_node_dim, n_features=n_features)

    def forward(self, x, edge_index, edge_attr, batch):
        connections_embedding = self.connection_embedding_layer(x, edge_index, edge_attr)
        x = self.node_init_embedding_layer(x)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        norm = deg.pow(-0.5)
        norm[norm == float('inf')] = 0
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, edge_attr=edge_attr, connections_embedding=connections_embedding,
                               norm=norm, k=k)
        avg_x = scatter(x, batch, dim=0, dim_size=batch.max() + 1, reduce="mean")\
            .gather(dim=0, index=batch.unsqueeze(-1).repeat(1, x.size(1)))
        avg_x_theta6 = self.theta6(avg_x)
        concat = torch.cat([avg_x_theta6, x], dim=1)
        return self.theta7(concat)

    def message(self, x_j, edge_attr):
        # x_j: [E, n_features]
        # edge_attr: [E, 1]
        return edge_attr * x_j

    def update(self, aggr_out, norm, connections_embedding, x, k):
        out1 = aggr_out * norm.view(-1, 1)
        out2 = torch.cat([out1, connections_embedding], dim=1)
        out3 = self.theta4layers[k](out2)
        out4 = self.theta5layers[k](torch.cat([x, out3], dim=1))
        return out4


if __name__ == '__main__':
    from utils.datagenerate import generate_graph_data_list
    max_graph_size = 49
    dataloader, distance_matrix = torch.load('./data/data_49.pt')
    mpnn = Mpnn(input_node_dim=max_graph_size, n_features=56, K=2)
    for data in dataloader:
        print(data.x.shape, data.edge_index.shape, data.edge_attr.shape)
        embeddings = mpnn(data.x, data.edge_index, data.edge_attr, data.batch)
        print(embeddings.shape)

# a,b  a <- b