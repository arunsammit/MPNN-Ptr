import torch
from typing import Tuple
from torch import Tensor, nn
import torch.nn.functional as F
import math
from gumbel import gumbel_like, gumbel_with_maximum

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, p):
        super(TransformerEncoder, self).__init__()
        # try to subclass nn.TransformerEncoderLayer in order to remove the bias in input/output projection layers by setting bias = False in the constructor of MultiheadAttention Module
        # do the above only if you suspect that it is causing any issues (but I don't think it will cause any issues)
        # also if you exactly want to replicate the wouter kool's approach you can replace the self.norm1 and self.norm2 with nn.BatchNorm1d
        encoder_layer = nn.TransformerEncoderLayer(d_model = input_dim, nhead = 8, dim_feedforward = hidden_dim, dropout = p)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
    def forward(self, input, mask):
        mask = mask == 0
        print(mask)
        node_embeddings = self.encoder(input, src_key_padding_mask = mask)
        graph_embeddings = torch.mean(node_embeddings, dim = 0)
        return node_embeddings, graph_embeddings

class TransformerAttention(nn.Module):
    def __init__(self, input_dim, logit_clipping = True, clip_value = 10):
        super(TransformerAttention, self).__init__()
        self.input_dim = input_dim
        # try with bias = True as well to check if it is 
        self.wq = nn.Linear(input_dim, input_dim, bias = False)
        self.wk = nn.Linear(input_dim, input_dim, bias = False)
        self.logit_clipping = logit_clipping
        self.clip_value = clip_value
    def forward(self, q:Tensor, k:Tensor, k_mask:Tensor = None):
        """
        q: (L, N, input_dim) where L is the target sequence length, N is the batch_size
        k: (S, N, input_dim) where S is the source sequence length
        k_mask: (N, S) which denotes which values will be ignored for the attention computation
        returns attn: (N, L, S)
        """
        L = q.size(0)
        # (L, N, input_dim) -> (L, N, input_dim)
        q = self.wq(q) / math.sqrt(self.input_dim)
        # (S, N, input_dim) -> (S, N, input_dim)
        k = self.wk(k)
        # (L, N, input_dim) X (S, N, input_dim) -> (N, L, S)
        attn = torch.einsum('lnd,snd -> nls', q, k)
        # attn = torch.bmm(q.permute(1,0,2), k.permute(1,2,0))
        if self.logit_clipping:
            attn = self.clip_value * torch.tanh(attn)
        if k_mask is not None:
            attn = attn.masked_fill(k_mask.unsqueeze(1).repeat(1, L, 1), float("-inf"))
        return attn

class TransformerPointerNet(nn.Module):
    def preprocess(self, input, mask) -> Tuple[Tensor, Tensor]:
        lengths = mask.sum(dim=1, dtype=torch.long)
        # seq_len is the length of the longest sequence in the batch
        seq_len = torch.max(lengths).item()
        # reshape input and mask to remove extra padding
        # first dimension of the input is seq_len
        input = input[:seq_len]
        # second dimension of the mask is seq_len
        mask = mask[:, :seq_len]
        return input, mask
    def __init__(self, input_dim, hidden_dim, n_layers, p, device, logit_clipping=True, num_heads = 8, decoding_type = 'sampling',):
        self.encoder: TransformerEncoder = TransformerEncoder(input_dim, hidden_dim, n_layers, p)
        self.v1 = nn.Parameter(torch.empty(1, input_dim, device=device))
        self.v2 = nn.Parameter(torch.empty(1, input_dim, device=device))
        # if model is not able to learn check with  bias = False once
        self.mha = nn.MultiheadAttention(input_dim, num_heads, dropout=p)
        self.attn = TransformerAttention(input_dim, logit_clipping = logit_clipping)
        self.decoding_type = decoding_type
    def forward(self, input, mask, num_samples = 1):
        input, mask = self.preprocess(input, mask)
        batch_size = input.size(1)
        seq_len = input.size(0)
        predicted_mappings = torch.zeros(batch_size, seq_len, seq_len, device=input.device)
        node_embeddings, graph_embeddings = self.encoder(input, mask)
        # shape of node_embeddings: (seq_len, batch_size, input_dim)
        # shape of graph_embeddings: (batch_size, input_dim)
        log_probs_sum = torch.zeros(batch_size * num_samples, dtype=torch.float32, device=input.device)
        first_decoded_embeddings = self.v1.repeat(batch_size, 1)
        prev_decoded_embeddings = self.v2.repeat(batch_size, 1)
        mask_decoding = mask.clone()
        for t in range(seq_len):
            # prepare the context embedding
            # shape of context_embedding: (batch_size, input_dim)
            context_embedding = torch.cat([graph_embeddings, prev_decoded_embeddings, first_decoded_embeddings], dim=1)
            context_embedding_glimpsed = self.mha(context_embedding.unsqueeze(1), node_embeddings, node_embeddings, key_padding_mask=mask_decoding)
            # shape of context_embedding: (batch_size, 1, input_dim)
            logits = self.attn(context_embedding_glimpsed, node_embeddings, mask_decoding)
            # shape of logits: (batch_size, 1, seq_len)
            logits = logits.squeeze(1)
            log_probs = F.log_softmax(logits, dim=-1)
            if t == 0:
                if self.decoding_type != 'sampling':
                    if self.decoding_type == 'sampling-w/o-replacement':
                        scores = log_probs + gumbel_like(log_probs)
                    elif self.decoding_type == 'greedy':
                        scores = log_probs
                    _, selected_indices = torch.topk(scores, min(num_samples, seq_len), dim=-1)
                    if num_samples > seq_len:
                        selected_indices = \
                            F.pad(selected_indices,(0, num_samples - seq_len),'constant', -1)
                    # shape of selected_indices: (batch_size, num_samples)
                else:
                    selected_indices = torch.multinomial(log_probs.exp(), num_samples, replacement=True).long()
                log_probs = log_probs.unsqueeze(1).repeat_interleave(num_samples, dim=1)
                # shape of log_probs: (batch_size, num_samples, seq_len)
                log_probs = log_probs.masked_fill((selected_indices == -1).unsqueeze(-1), float("-inf"))
                if self.decoding_type == 'sampling-w/o-replacement':
                    scores = scores.unsqueeze(1).repeat_interleave(num_samples, dim=1)
                    scores = scores.masked_fill((selected_indices == -1).unsqueeze(-1), float("-inf"))
                mask_decoding = mask_decoding.unsqueeze(1).repeat_interleave(num_samples, dim=1)


                




            








        
    
def main():
    torch.manual_seed(100)
    import random
    random.seed(100)
    input_dim = 16
    n_layers = 2
    hidden_dim = 32
    p = 0
    batch_size = 2
    seq_len = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = TransformerEncoder(input_dim, hidden_dim, n_layers, p).to(device)
    input = torch.randn(seq_len, batch_size, input_dim).to(device)
    mask = torch.ones(batch_size, seq_len).to(device)
    mask[:, -1:] = 0
    print(input)
    input.permute(1,0,2)[mask == 0] = -5
    print(input)
    output = encoder(input, mask)
    print(output)
if __name__ == '__main__':
    main()