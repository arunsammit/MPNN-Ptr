from torch import Tensor, nn
import torch
class Encoder:
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers,
                 dropout, pad_idx):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embedding_dim,
                                      padding_idx=pad_idx)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                           dropout=dropout, bidirectional=True)