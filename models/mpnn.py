import torch
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


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
        # x: [N, in_channels]
        # edge_index: [2, E]
        # print(aggr_out.shape, edge_index.shape)
        row, col = edge_index
        deg = degree(col, aggr_out.size(0), dtype=aggr_out.dtype)
        return self.layer3(torch.concat([aggr_out, deg.unsqueeze(-1)], dim=1))

    def forward(self, x, edge_index, edge_attr):
        # x: [N, in_channels]
        # edge_index: [2, E]
        # edge_attr: [E, 1]
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)


class Mpnn(MessagePassing):
    def __init__(self, in_channels, n_features, K):
        # in_channels: dimension of input node features.
        # n_features: dimension of the output embeddings
        super().__init__(aggr='add')
        self.K = K
        self.node_init_embedding_layer = nn.Sequential(
            nn.Linear(in_channels, n_features, bias=False),
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
        self.connection_embedding_layer = ConnectionsEmbedding(in_channels=in_channels, n_features=n_features)

    def forward(self, x, edge_index, edge_attr, batch=None):
        connections_embedding = self.connection_embedding_layer(x, edge_index, edge_attr)
        x = self.node_init_embedding_layer(x)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        norm = deg.pow(-0.5)
        norm[norm == float('inf')] = 0
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, edge_attr=edge_attr, connections_embedding=connections_embedding,
                               norm=norm, k=k)
        avg_x = x.mean(dim=0)
        avg_x_theta6 = self.theta6(avg_x.unsqueeze(0)).expand(x.shape[0], -1)
        concat = torch.cat([avg_x_theta6, x], dim=1)
        return self.theta7(concat)

    def message(self, x_j, edge_attr):
        # x_j: [E, n_features]
        # edge_attr: [E, 1]
        return edge_attr * x_j

    def update(self, aggr_out, norm, connections_embedding, x, k):
        out1 = aggr_out * norm.view(-1, 1)
        out2 = torch.concat([out1, connections_embedding], dim=1)
        out3 = self.theta4layers[k](out2)
        out4 = self.theta5layers[k](torch.concat([x, out3], dim=1))
        return out4
