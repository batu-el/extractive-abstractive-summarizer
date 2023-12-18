#####################################################################################
'''                                                                                 
# Minimum Bayes Risk Decoding
# based on Follow the Wisdom of the Crowd: Effective Text Generation via Minimum Bayes Risk Decoding
# References:   https://aclanthology.org/2023.findings-acl.262.pdf
#               
'''
#####################################################################################


import torch
import config
from utils.masks import generate_square_subsequent_mask
from decoders.temperature import temperature_decode

def candidates_set(summarizer, src, src_mask, max_len, start_symbol, num_samples, temperature):
    # generate a set of num_samples candidates using temperature decoding
    candidates = {}
    for i in range(num_samples):
        # decode a target sequence using temperature
        tgt_tokens = temperature_decode(summarizer, src, src_mask, max_len, start_symbol, temperature).flatten()
        # convert it to a string
        tgt_translation = " ".join(summarizer.vocab_transform.lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")
        # add it to the dict
        candidates[i] = tgt_translation 
    return candidates

import datasets
import bert_score
import numpy as np

def mbr_decode(summarizer, src, src_mask, max_len, start_symbol, num_samples=10, temperature=1):
    # generate the candidates set
    candidates = candidates_set(summarizer, src, src_mask, max_len, start_symbol, num_samples, temperature)
    # get the candidates in a list
    candidates = list(candidates.values())
    # load the BERTScore metric for evaluating similarity
    bertscore = datasets.load_metric("bertscore")
    # score candidate pairs with BERTScore
    score_matrix = np.zeros((num_samples, num_samples))
    for j1, cand1 in enumerate(candidates):
        for j2, cand2 in enumerate(candidates):
            # compare every pair of candidates only once
            if j1 < j2:
                # compute the BERTScore F1 score for the pair of candidates
                score = bertscore.compute(predictions=[cand1], references=[cand2], lang='en')['f1'][0]
                # store the score in the matrix symmetrically
                score_matrix[j1][j2] = score_matrix[j2][j1] = score
    # get candidate scores
    sum_scores = np.sum(score_matrix, axis=1)
    # get the candidate with max score ( maximum expected utility and minimum bayes risk are the same by duality)
    index = np.argmax(sum_scores)
    # return the best candidate
    final_output = candidates[index]
    return final_output

# Same function but verbose - returns all candidates and the best candidate index instead
# I used it to inspect the eliminated candidates
def mbr_decode_verbose(summarizer, src, src_mask, max_len, start_symbol, num_samples=10, temperature=1):
    candidates = candidates_set(summarizer, src, src_mask, max_len, start_symbol, num_samples, temperature)
    candidates = list(candidates.values())
    bertscore = datasets.load_metric("bertscore")
    score_matrix = np.zeros((num_samples, num_samples))
    for j1, cand1 in enumerate(candidates):
        for j2, cand2 in enumerate(candidates):
            if j1 < j2:
                score = bertscore.compute(predictions=[cand1], references=[cand2], lang='en')['f1'][0]
                score_matrix[j1][j2] = score_matrix[j2][j1] = score
    sum_scores = np.sum(score_matrix, axis=1)
    index = np.argmax(sum_scores)
    final_output = candidates[index]
    return candidates, index