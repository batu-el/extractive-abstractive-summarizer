#####################################################################################
'''                                                                                 
# Greedy Decoding
# based on Pytorch Tutorials
# References:   https://pytorch.org/tutorials/beginner/translation_transformer.html
#               
'''
#####################################################################################

import torch
import config
from utils.masks import generate_square_subsequent_mask

def greedy_decode( summarizer, src, src_mask, max_len, start_symbol):
    # get the model
    model = summarizer.model
    # put the source sentence and mask to device 
    src = src.to(config.DEVICE)
    src_mask = src_mask.to(config.DEVICE)
    # encode the source sentence using encoder
    memory = model.encode(src, src_mask)
    # initialize the first token of the output using start symbol
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(config.DEVICE)

    for i in range(max_len-1):
        memory = memory.to(config.DEVICE)
        # create target mask
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(config.DEVICE)
        # decode
        out = model.decode(ys, memory, tgt_mask)
        # transpose
        out = out.transpose(0, 1)
        # output of the linear layer
        prob = model.generator(out[:, -1])
        # select the token with highest score
        # this score is not probability since it is not yet passed through a softmax
        # but since softmax is monotonic, just taking the argmax at this step gives the same result
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        # add predicted word to the output
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        # end when EOS is generated
        if next_word == config.EOS_IDX:
            break
    return ys