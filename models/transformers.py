import torch.nn as nn
import torch

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, p):
        encoder_layer = nn.TransformerEncoderLayer(d_model = input_dim, nhead = 8, dim_feedforward = hidden_dim, dropout = p)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
    def forward(self, input, mask):
        return self.encoder(input, mask)