from re import L
from numpy.lib.twodim_base import diag
import torch
from typing import Tuple
from torch import Tensor, nn
import torch.nn.functional as F
import math
from gumbel import gumbel_like, gumbel_with_maximum
from models.transformers import rearrange, TransformerAttention
from utils.datagenerate import NocNumberingNew

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim = 128, dim_feedforward = 512, n_layers = 3, p=0):
        super(TransformerEncoder, self).__init__()
        # try to subclass nn.TransformerEncoderLayer in order to remove the bias in input/output projection layers by setting bias = False in the constructor of MultiheadAttention Module
        # do the above only if you suspect that it is causing any issues (but I don't think it will cause any issues)
        # also if you exactly want to replicate the wouter kool's approach you can replace the self.norm1 and self.norm2 with nn.BatchNorm1d
        encoder_layer = nn.TransformerEncoderLayer(d_model = hidden_dim, nhead = 8, dim_feedforward = dim_feedforward, dropout = p)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            # nn.ReLU()
        )
    def forward(self, input, mask):
        # print(mask)
        input_proj = self.proj(input)
        node_embeddings = self.encoder(input_proj, src_key_padding_mask = mask)
        graph_embeddings = torch.mean(node_embeddings, dim = 0)
        return node_embeddings, graph_embeddings

class TransformerPointerNet2(nn.Module):
    "Transformer pointer net with modified decoding context and Graph Attention Network as the Encoder"
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
    def __init__(self, input_dim, hidden_dim, n_layers, p, device, logit_clipping=True, num_heads = 8, decoding_type = 'sampling', numbering = NocNumberingNew()):
        print("using transformers_v2")
        super(TransformerPointerNet2, self).__init__()
        self.encoder: TransformerEncoder = TransformerEncoder(input_dim, hidden_dim=hidden_dim, n_layers=n_layers, p=p)
        self.v1 = nn.Parameter(torch.empty(1, hidden_dim, device=device))
        self.v2 = nn.Parameter(torch.empty(1, hidden_dim, device=device))
        # try to add non-linearity (nn.ReLU) & bias to improve the performance to improve the performance 
        self.proj1 = nn.Sequential( 
            nn.Linear(3*hidden_dim, hidden_dim, bias = False),
            # nn.ReLU()
        )
        # if model is not able to learn check with  bias = False once
        self.mha = nn.MultiheadAttention(hidden_dim, num_heads, dropout=p)
        self.attn = TransformerAttention(hidden_dim, logit_clipping = logit_clipping)
        self.decoding_type = decoding_type
        self.numbering = numbering
    def forward(self, input, mask, num_samples = 1):
        input, mask = self.preprocess(input, mask)
        # shape of mask: (batch_size, seq_len)
        # shape of input: (seq_len, batch_size, hidden_dim)
        batch_size = input.size(1)
        seq_len = input.size(0)
        predicted_mappings = torch.zeros(batch_size, num_samples, seq_len, device=input.device, dtype = torch.long)
        node_embeddings, graph_embeddings = self.encoder(input, mask == 0)
        # shape of node_embeddings: (seq_len, batch_size, hidden_dim)
        # shape of graph_embeddings: (batch_size, hidden_dim)
        log_probs_sum = torch.zeros(batch_size, num_samples, dtype=torch.float32, device=input.device)
        adjacent_decoded_embeddings = self.v1.repeat(batch_size, 1)
        prev_decoded_embeddings = self.v2.repeat(batch_size, 1)
        mask_decoding = mask.clone()
        mask_multiple_samples = mask.unsqueeze(1).expand(-1, num_samples, -1)
        # shape of mask_multiple_samples: (batch_size, num_samples, seq_len)
        for t in range(seq_len):
            # prepare the context embedding
            # shape of context_embedding: (batch_size, hidden_dim)
            # mask_decoding = mask_decoding.view(batch_size * num_samples, -1)
            if t > 1:
                adj_idx = self.numbering.get_adj_idx(t   )
                adjacent_indices = predicted_mappings[:,:,adj_idx].unsqueeze(-1).clone()
                adjacent_indices = adjacent_indices.masked_fill(adjacent_indices == -1, 0)
                adjacent_decoded_embeddings = node_embeds_batch_first.gather(1, adjacent_indices.expand(-1,-1,hidden_dim))\
                .view(batch_size * num_samples, -1)
            context_embedding = torch.cat([graph_embeddings, prev_decoded_embeddings, adjacent_decoded_embeddings], dim=1)
            context_embedding_proj = self.proj1(context_embedding.unsqueeze(0))            
            context_embedding_glimpsed, _ = self.mha(context_embedding_proj, node_embeddings, node_embeddings, key_padding_mask=mask_decoding == 0, need_weights=False)
            # shape of context_embedding_proj: (batch_size, 1, hidden_dim)
            logits = self.attn(context_embedding_glimpsed, node_embeddings, mask_decoding == 0).view(batch_size, -1, seq_len)
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
                mask_decoding = mask_decoding.repeat_interleave(num_samples, dim=0)
                graph_embeddings = graph_embeddings.repeat_interleave(num_samples, dim=0)
                node_embeddings = node_embeddings.repeat_interleave(num_samples, dim=1)
                # filling mask with -1 is not required
                adjacent_decoded_embeddings = adjacent_decoded_embeddings.repeat_interleave(num_samples, dim=0)
            else:
                if self.decoding_type != 'sampling':
                    if self.decoding_type == 'sampling-w/o-replacement':
                        scores, _ = gumbel_with_maximum(log_probs + log_probs_sum.unsqueeze(-1), g_log_probs)
                    elif self.decoding_type == 'greedy':
                        scores = log_probs + log_probs_sum.unsqueeze(-1)
                    # scores shape: (batch_size, num_samples, seq_len)
                    scores_per_batch = scores.view(batch_size, -1)
                    _, indices_buf = torch.topk(scores_per_batch, num_samples, dim=-1)
                    beams_buf = torch.div(indices_buf, seq_len, rounding_mode="floor")
                    # shape of indices_buf: (batch_size, num_samples)
                    indices_buf = indices_buf.fmod(seq_len)
                    selected_indices = indices_buf
                    predicted_mappings = rearrange(predicted_mappings, beams_buf)
                    log_probs = rearrange(log_probs, beams_buf)
                    
                    if self.decoding_type == 'sampling-w/o-replacement':
                        scores = rearrange(scores, beams_buf)
                else:
                    # shape of log_probs: (batch_size, num_samples, seq_len)
                    selected_indices = torch.multinomial(log_probs.exp().view(-1, seq_len), 1, replacement=True).long().squeeze(-1).view(batch_size, num_samples)
            # generating the decoding context for the next step
            hidden_dim = node_embeddings.size(-1)
            predicted_mappings[:, :, t] = selected_indices
            node_embeds_batch_first = node_embeddings.transpose(0, 1)
            # creating the indices for gathering log_probs corresponding to choosen indices
            gather_indices = selected_indices.unsqueeze(-1).clone()
            gather_indices = gather_indices.masked_fill(gather_indices == -1, 0)
            prev_decoded_embeddings = node_embeds_batch_first.gather(1, gather_indices.expand(-1,-1, hidden_dim))\
                .view(batch_size * num_samples, -1)
            # below implementation is written assuming that all graphs in the given batch are of same size
            curr_log_probs = log_probs.gather(-1, gather_indices).squeeze(-1) * mask_multiple_samples[:,:, t]
            log_probs_sum += curr_log_probs
            if self.decoding_type == 'sampling-w/o-replacement':
                g_log_probs = scores.gather(-1, gather_indices).squeeze(-1) * mask_multiple_samples[:,:, t]
            # making the mask 0 corresponding to the
            mask_decoding.scatter_(-1, gather_indices.view(batch_size * num_samples, -1), 0)

        predicted_mappings = predicted_mappings.masked_fill(mask_multiple_samples == 0, -1)
        # changing the shapes so that the are all nth samples are present together
        predicted_mappings = predicted_mappings.transpose(0, 1).reshape(-1, seq_len)
        log_probs_sum = log_probs_sum.transpose(0, 1).reshape(-1)
        if self.decoding_type == 'sampling-w/o-replacement':
            g_log_probs = g_log_probs.transpose(0, 1).reshape(-1)
            return predicted_mappings, log_probs_sum, g_log_probs
        else:
            return predicted_mappings, log_probs_sum
        
    
def main():
    torch.manual_seed(100)
    import random
    random.seed(100)
    input_dim = 16
    n_layers = 3
    hidden_dim = 32
    p = 0
    batch_size = 2
    seq_len = 10
    num_samples = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # encoder = TransformerEncoder(input_dim, hidden_dim, n_layers, p).to(device)
    tPtrNet = TransformerPointerNet2(input_dim, hidden_dim, n_layers, p, device, logit_clipping=True).to(device)
    from utils.utils import init_weights
    tPtrNet.apply(init_weights)
    input = torch.randn(seq_len, batch_size, input_dim).to(device)
    mask = torch.ones(batch_size, seq_len).to(device)
    # mask[:, -1:] = 0
    # print(input)
    # input.permute(1,0,2)[mask == 0] = -5
    print(input)
    tPtrNet.decoding_type = 'sampling'
    mappings,  ll_sum = tPtrNet(input, mask, num_samples)
    print(mappings.view(num_samples,batch_size, seq_len).transpose(0,1))
    print(ll_sum.view(num_samples,batch_size).transpose(0,1))
if __name__ == '__main__':
    main()