import sys
sys.path.append('../')

import config
import torch

''' MASKING UTILS '''
def generate_square_subsequent_mask(size):
    # upper triangular matrix of ones with given size
    # elements above the main diagonal are ones
    mask = (torch.triu(torch.ones((size, size), device=config.DEVICE)) == 1).transpose(0, 1)
    # convert the boolean mask to float
    # replace 0s with negative infinity and 1s with 0
    # -inf masks out subsequent positions during the attention calculation,
    # so that the model can't see the next tokens in target sequence
    # this mask is used in decoder
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
def create_mask(src, tgt):
    # length of source sequence
    src_seq_len = src.shape[0]
    # length of target sequence
    tgt_seq_len = tgt.shape[0]
    # causal mask to prevent attending to subsequent positions
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    # no causal mask for the source sequence since encoder attends to all the source sequence
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=config.DEVICE).type(torch.bool)
    # padding masks so that the model does not use these positions for calculating self-attention
    src_padding_mask = (src == config.PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == config.PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask