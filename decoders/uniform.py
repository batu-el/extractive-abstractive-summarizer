#####################################################################################
'''                                                                                 
# Uniform Random Decoding
#               
'''
#####################################################################################
import torch
import config
from utils.masks import generate_square_subsequent_mask

# function to generate output sequence using greedy algorithm
def uniform_decode(summarizer, src, src_mask, max_len, start_symbol, temperature=0.5):
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
        tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(config.DEVICE)
        # decode
        out = model.decode(ys, memory, tgt_mask)
        # transpose
        out = out.transpose(0, 1)
        # output of the linear layer
        prob = model.generator(out[:, -1]).detach() # the output of the linear layer before the softmax
        # replace the prob by ones to ignore the model probability
        prob = torch.ones_like(prob)
        # pass it though softmax
        softmax = torch.nn.Softmax()
        temperature_prob = prob 
        dist = softmax(temperature_prob)
        # sample using multinomial where every option has equal probability
        next_word = torch.multinomial(dist, 1)[0]
        next_word = next_word.item()
        # add next word to the sequence
        ys = torch.cat([ys,torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        # beak if the next word is EOS token
        if next_word == config.EOS_IDX:
            break
    return ys