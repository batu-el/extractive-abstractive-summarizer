#####################################################################################
'''                                                                                 
# Temperature Decoding
# based on Controlling Linguistic Style Aspects in Neural Language Generation
# References:   https://arxiv.org/abs/1707.02633
#               
'''
#####################################################################################

import torch
import config
from utils.masks import generate_square_subsequent_mask

def temperature_decode(summarizer, src, src_mask, max_len, start_symbol, temperature=0.5):
    # get the model
    model = summarizer.model
    # put source and target to device
    src = src.to(config.DEVICE)
    src_mask = src_mask.to(config.DEVICE)
    # encode the source sequence
    memory = model.encode(src, src_mask)
    # begin the sequence with the start symbol
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(config.DEVICE)
    for i in range(max_len-1):
        # put the context to device
        memory = memory.to(config.DEVICE)
        # generate target mask to hide next
        tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(config.DEVICE)
        # decode to get the decoder output
        out = model.decode(ys, memory, tgt_mask)
        # transpose it
        out = out.transpose(0, 1)
        # get the output of linear layer
        prob = model.generator(out[:, -1]).detach() # the output of the linear layer before the softmax
        softmax = torch.nn.Softmax()
        # scale the linear layer output dividing by temperature
        temperature_prob = prob / temperature
        # pass this through softmax to get the distribution
        dist = softmax(temperature_prob)
        # sample the next word from the multinomial dist using these probabilities
        next_word = torch.multinomial(dist, 1)[0]
        next_word = next_word.item()
        # add token to the sequence
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        # end if next word is token is EOS token
        if next_word == config.EOS_IDX:
            break
    return ys