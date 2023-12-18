#####################################################################################
'''                                                                                 
# Beam Search
# based on "Transformers and Pretrained Language Models" chapter in Jurafky and Martin
# different from their implementation, I implemented a recursive Beam Search
# References:   https://web.stanford.edu/~jurafsky/slp3/10.pdf
#               
'''
#####################################################################################

import torch
import config
from utils.masks import generate_square_subsequent_mask

''' BEAM EXPANSION '''
def beam_expansion(beam , memory, model, beam_width , src):
    # the list to store the expanded beam sequences
    expanded_beam = []
    # set memory configuration
    memory = memory.to(config.DEVICE)

    for sequence in beam:
        # if the last token of the sequence is the End Of Sequence (EOS) token the sequence is complete, add back
        if sequence[1][-1] == config.EOS_IDX:
            expanded_beam.append(sequence)
        else:
            # if incomplete generate a mask to hide future tokens
            tgt_mask = (generate_square_subsequent_mask(sequence[1].size(0)).type(torch.bool)).to(config.DEVICE) # just false
            # decode
            out = model.decode(sequence[1], memory, tgt_mask)
            # transpose
            out = out.transpose(0, 1)
            # get the output of the linear layer before the softmax
            prob = model.generator(out[:, -1]).detach() 
            # apply softmax to convert logits to probabilities and take the log
            softmax = torch.nn.Softmax()
            log_prob = softmax(prob).log()
            # select top beam_width next tokens
            scores, next_words = torch.topk(log_prob, beam_width, dim=1)
            # expand with these top k next tokens
            for i in range(beam_width):
                new_score = sequence[0] + scores[0][i]
                new_ys = torch.cat([sequence[1], torch.ones(1, 1).type_as(src.data).fill_(next_words[0][i].item())], dim=0)
                expanded_beam.append((new_score , new_ys))
    # return expanded beam
    return expanded_beam

''' BEAM PRUNING '''
def beam_pruning(beam, beam_width ):
    # choose top beam_width scorers
    beam_scores = [beam[i][0] for i in range(len(beam))]
    topk_idx = torch.topk(torch.tensor(beam_scores) , beam_width)[1]
    pruned_beam = [beam[i] for i in topk_idx]
    return pruned_beam

''' BEAM SEARCH '''
def beam_search(beam,memory, model, beam_width, max_len, src):
    # recursive beam search
    complete = True
    for seq in beam:
        lst_token = seq[1][-1]
        seq_len = len(seq[1])
        if (lst_token != config.EOS_IDX) and (seq_len < max_len):
            # if there is a sequence in the beam whose last token is not the EOS token
            # and the length of that sequence is less than the maximum length
            # then the beam search is not complete
            complete = False
            break
    if complete:
        # prune the beam to be of width 1
        return beam_pruning(beam, 1)
    else:
        # expand the beam
        beam = beam_expansion(beam , memory, model, beam_width, src)
        # prune the beam
        beam = beam_pruning(beam, beam_width )
        # call beamseach with the new beam
        return beam_search(beam,memory, model, beam_width, max_len, src)

''' BEAM SEARCH DECODE '''
def beam_search_decode(summarizer, src, src_mask, max_len, start_symbol, beam_width=5):
    # get the model
    model = summarizer.model
    # put the source sentence and mask to device 
    src = src.to(config.DEVICE)
    src_mask = src_mask.to(config.DEVICE)
    # encode the source sentence using encoder
    memory = model.encode(src, src_mask) 

    ''' BEAM INITIALIZATION '''
    # initialize the first token of the output using start symbol
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(config.DEVICE)
    ys_score = 0
    # initialize the beam with one sequence that contains the start symbol and has likelihood 0
    beam = [(ys_score , ys)]
    # get the beam search output
    result = beam_search(beam, memory, model, beam_width, max_len, src)
    # get the log-likelihood and the sequence
    #log_likelihood = result[0][0] # to be used for tests
    sequence = result[0][1]
    # return the sequence
    return sequence