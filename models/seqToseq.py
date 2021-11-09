from torch import Tensor, nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers,
                 p):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(input_dim, hidden_dim, n_layers,
                           dropout=p)

    def forward(self, input:PackedSequence):
        # input shape: (seq_len, batch, input_dim)
        # input_lengths shape: (batch)
        # output shape: (seq_len, batch, hidden_dim)
        packed_outputs, (hidden, cell) = self.rnn(input)
        output, _ = pad_packed_sequence(packed_outputs)
        # hidden shape: (n_layers, batch, hidden_dim) cell shape: (n_layers, batch, hidden_dim)
        # if encoder is bidirectional
        # pass hidden and cell through a linear layer to match decoder hidden and cell dimension
        return output, (hidden, cell)


class Attention(nn.Module):
    def __init__(self, enc_dim, dec_dim):
        super(Attention, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(enc_dim + dec_dim, dec_dim),
            nn.Tanh(),
        )
        self.v = nn.Linear(dec_dim, 1, bias=False)

    def forward(self, decoder_output, encoder_outputs, mask):
        # decoder_output is the output of the decoder of single step
        # encoder_outputs is a list of all the encoder outputs
        # decoder_output shape: (batch, dec_dim)
        # encoder_outputs shape: (seq_len, batch, enc_dim)
        seq_len = encoder_outputs.shape[0]
        batch_size = encoder_outputs.shape[1]
        decoder_output = decoder_output.unsqueeze(1).repeat(1, seq_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # decoder_output shape: (batch, seq_len, dec_dim)
        # encoder_outputs shape: (batch, seq_len, enc_dim)
        energy = self.attn(torch.cat((decoder_output, encoder_outputs), dim=2))
        # energy shape: (batch, seq_len, dec_dim)
        attention = self.v(energy).squeeze(2)
        # use mask to remove the attention weights for padded values
        attention = attention.masked_fill(mask == 0, -1e10)
        # attention shape: (batch, seq_len)
        return attention


class Decoder(nn.Module):
    # module for single step of decoding process
    def __init__(self, input_dim, hidden_dim, n_layers, p):
        super(Decoder, self).__init__()
        self.output_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=p)

    def forward(self, input, hidden, cell):
        # input shape: (batch, input_dim)
        # hidden shape: (n_layers, batch, hidden_dim)
        # encoder_outputs shape: (seq_len, batch, enc_dim*2)
        # output shape: (batch, input_dim)
        input = input.unsqueeze(0)
        output, (hidden, cell) = self.rnn(input, (hidden, cell))
        output = output.squeeze(0)
        return output, (hidden, cell)


class PointerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, p):
        super().__init__()
        self.encoder:Encoder = Encoder(input_dim, hidden_dim, n_layers, p)
        self.decoder:Decoder = Decoder(input_dim, hidden_dim, n_layers, p)
        self.attn:Attention = Attention(hidden_dim, hidden_dim)
        self.initial_decoder_input = nn.Parameter(torch.zeros(1, input_dim))

    def forward(self, input:Tensor, mask):
        # input shape: (seq_len, batch, input_dim)
        batch_size = input.size(1)
        # seq_len is the length of the longest sequence in the batch
        seq_len = input.size(0)
        # check the dimension of output
        outputs = torch.zeros(seq_len, batch_size, seq_len)
        lengths = mask.sum(dim=1)
        packed_embeddings = torch.nn.utils.rnn.pack_padded_sequence(input, lengths.to('cpu'),
                                                                    enforce_sorted=False)
        encoder_outputs, (hidden, cell) = self.encoder(packed_embeddings)

        # first input should be a part of model learnable parameters
        decoder_input = self.initial_decoder_input.repeat(batch_size, 1)
        for t in range(1, seq_len):
            output, (hidden, cell) = self.decoder(decoder_input, hidden, cell)
            outputs[t] = self.attn(output, encoder_outputs, mask)
            pointed_inputs = input.data[outputs[t].argmax(dim=1), torch.arange(batch_size)]
            # pointed_inputs shape: (batch, input_dim)
            decoder_input = pointed_inputs
