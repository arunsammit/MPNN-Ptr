from torch import Tensor, nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers,
                 p):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(input_dim, hidden_dim, n_layers,
                           dropout=p)

    def forward(self, input, mask):
        # input shape: (seq_len, batch, input_dim)
        # mask shape: (batch, seq_len)
        lengths = mask.sum(dim=1)
        # lengths shape: (batch)
        packed_inputs = torch.nn.utils.rnn.pack_padded_sequence(input, lengths.to('cpu'),
                                                                enforce_sorted=False)
        packed_outputs, (hidden, cell) = self.rnn(packed_inputs)
        output, _ = pad_packed_sequence(packed_outputs)
        # output shape: (seq_len, batch, hidden_dim)
        # hidden shape: (n_layers, batch, hidden_dim) cell shape: (n_layers, batch, hidden_dim)
        # if encoder is bidirectional
        # pass hidden and cell through a linear layer to match decoder hidden and cell dimension
        seq_len = input.shape[0]
        # reshape output to (seq_len, batch, hidden_dim)
        return output, (hidden, cell)


class Attention(nn.Module):
    def __init__(self, enc_dim, dec_dim, logit_clipping=True, clip_value=10):
        super(Attention, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(enc_dim + dec_dim, dec_dim),
            nn.Tanh(),
        )
        self.v = nn.Linear(dec_dim, 1, bias=False)
        self.logit_clipping = logit_clipping
        self.clip_value = clip_value

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
        if self.logit_clipping:
            attention = self.clip_value * torch.tanh(attention)
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
    def __init__(self, input_dim, hidden_dim, n_layers, p, device):
        super().__init__()
        self.encoder: Encoder = Encoder(input_dim, hidden_dim, n_layers, p)
        self.decoder: Decoder = Decoder(input_dim, hidden_dim, n_layers, p)
        self.attn: Attention = Attention(hidden_dim, hidden_dim)
        self.initial_decoder_input = nn.Parameter(torch.zeros(1, input_dim))
        self.device = device

    def preprocess(self, input, mask):
        lengths = mask.sum(dim=1, dtype=torch.long)
        # seq_len is the length of the longest sequence in the batch
        seq_len = torch.max(lengths).item()
        # reshape input and mask to remove extra padding
        input = input[:seq_len]
        mask = mask[:, :seq_len]
        batch_size = input.size(1)
        return input, mask

    def forward(self, input: Tensor, mask):
        # input shape: (seq_len, batch, input_dim)
        # mask shape: (batch, seq_len)
        input, mask = self.preprocess(input, mask)
        batch_size = input.size(1)
        seq_len = input.size(0)
        # Tensor to store the predicted mapping
        # predicted_mappings shape: (batch, seq_len)
        predicted_mappings = torch.zeros(batch_size, seq_len, dtype=torch.int64).to(self.device)
        encoder_outputs, (hidden, cell) = self.encoder(input, mask)
        # first input should be a part of model learnable parameters
        decoder_input = self.initial_decoder_input.repeat(batch_size, 1)
        # mask to be used while calculating attention weights
        mask_decoding = mask.clone()
        log_probs_list = []
        for t in range(seq_len):
            output, (hidden, cell) = self.decoder(decoder_input, hidden, cell)
            logits = self.attn(output, encoder_outputs, mask_decoding)
            # logits shape: (batch, seq_len)
            log_probs = F.log_softmax(logits, dim=1)
            # log_probs shape: (batch, seq_len)
            selected_indices = torch.multinomial(log_probs.exp(), 1).long().squeeze(1)
            # selected_indices shape: (batch_size,)
            predicted_mappings[:, t] = selected_indices
            log_probs_list.append(log_probs)
            # decoder_input shape: (batch, input_dim)
            decoder_input = input[selected_indices, torch.arange(batch_size)]
            # update the mask_decoding to remove the pointed inputs
            mask_decoding.scatter_(1, selected_indices.unsqueeze(1), 0)
        # calculate the log likelihoods
        log_probs = torch.stack(log_probs_list, dim=1).to(self.device)
        # log_probs shape: (batch, seq_len, seq_len)
        log_likelihoods = log_probs.gather(2, predicted_mappings.unsqueeze(-1)).squeeze(-1)
        # log_likelihoods shape: (batch, seq_len)
        # ignoring the probabilities of the padded values
        log_likelihoods = log_likelihoods.masked_fill(mask == 0, 0)
        # log_likelihoods shape: (batch,)
        # assign -1 to the mappings corresponding to the padded values to denote that it is invalid
        predicted_mappings = predicted_mappings.masked_fill(mask == 0, -1)
        log_likelihoods_sum = log_likelihoods.sum(dim=1)
        return predicted_mappings, log_likelihoods_sum

    @torch.no_grad()
    def sample_multiple_mappings(self, input: Tensor, mask, n_samples):
        # sample n_samples mapping solutions for each sequence in the batch
        input, mask = self.preprocess(input, mask)
        batch_size = input.size(1)
        seq_len = input.size(0)
        predicted_mappings = torch.zeros(batch_size * n_samples, seq_len, dtype=torch.int64).to(self.device)
        encoder_outputs, (hidden, cell) = self.encoder(input, mask)
        decoder_input = self.initial_decoder_input.repeat(batch_size, 1)
        mask_decoding = mask.clone()
        for t in range(seq_len):
            output, (hidden, cell) = self.decoder(decoder_input, hidden, cell)
            if t == 0:
                logits = self.attn(output, encoder_outputs, mask_decoding)
                log_probs = F.log_softmax(logits, dim=1)
                # selected_indices shape: (batch, n_samples)
                selected_indices = torch.multinomial(log_probs.exp(), n_samples).long()
                predicted_mappings[:, t] = selected_indices.view(-1)
                mask_decoding = mask_decoding.repeat_interleave(n_samples, dim=0).clone()
                mask_decoding.scatter_(1, selected_indices.view(-1).unsqueeze(1), 0)
                hidden = hidden.repeat_interleave(n_samples, dim=1)
                cell = cell.repeat_interleave(n_samples, dim=1)
            else:
                logits = self.attn(output, encoder_outputs.repeat_interleave(n_samples, dim=1), mask_decoding)
                log_probs = F.log_softmax(logits, dim=1)
                selected_indices = torch.multinomial(log_probs.exp(), 1).long().squeeze(1)
                predicted_mappings[:, t] = selected_indices
                mask_decoding.scatter_(1, selected_indices.unsqueeze(1), 0)
            decoder_input = input.repeat_interleave(n_samples, dim=1)[selected_indices.view(-1),
                                                                      torch.arange(batch_size * n_samples)]
        predicted_mappings = predicted_mappings.masked_fill(mask.repeat_interleave(n_samples, dim=0) == 0, -1)
        # reshape the predicted_mappings to (batch_size, seq_len, n_samples)
        predicted_mappings = predicted_mappings.view(batch_size, n_samples, seq_len)

        return predicted_mappings


if __name__ == '__main__':
    # test the model
    input_dim = 5
    hidden_dim = 8
    n_layers = 2
    p = 0.1
    batch_size = 4
    max_seq_len = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PointerNet(input_dim, hidden_dim, n_layers, p, device)
    input = torch.randn(max_seq_len, batch_size, input_dim)
    mask = torch.tril(torch.ones(batch_size, max_seq_len))
    mask[-1, :] = 1
    predicted_mappings, log_likelihoods_sum = model(input, mask)
    samples = model.sample_multiple_mappings(input, mask, n_samples=2)
    print(samples)
    print(predicted_mappings)
    print(log_likelihoods_sum)
