# Import required modules and functions
from rouge_metric import PyRouge
import itertools
import numpy as np
import random

# lambda function for evaluating summary alignment using ROUGE-1
SummaryAlignment_R1 = lambda summary, article: PyRouge(rouge_n=(1)).evaluate([article[sentence_index] for sentence_index in summary], [article for i in range(len(summary))])['rouge-1']['f']

# The following commented out lines are alternative lambda functions for evaluating summary alignment using ROUGE-2 and ROUGE-4, which are currently not in use.
# SummaryAlignment_R2 = lambda summary, article: PyRouge(rouge_n=(2)).evaluate([article[sentence_index] for sentence_index in summary], [article for i in range(len(summary))])['rouge-2']['f']
# SummaryAlignment_R4 = lambda summary, article: PyRouge(rouge_n=(4)).evaluate([article[sentence_index] for sentence_index in summary], [article for i in range(len(summary))])['rouge-4']['f']

# lambda function for evaluating summary alignment using ROUGE-L
SummaryAlignment_RL = lambda summary, article: PyRouge(rouge_l=True).evaluate([article[sentence_index] for sentence_index in summary], [article for i in range(len(summary))])['rouge-l']['f']

# lambda function for calculating the summary alignment score
SummaryAlignment_1 = lambda summary, article: SummaryAlignment_R1(summary, article) 
# lambda function for calculating the sum of summary alignment scores for each sentence in the article
SummaryAlignment_2 = lambda summary, article: sum([SummaryAlignment_R1(summary, sentence) for sentence in article])
# lambda function for calculating the summary alignment score
SummaryAlignment_3 = lambda summary, article: SummaryAlignment_R1(summary, article) + SummaryAlignment_RL(summary, article) 

# Using SummaryAlignment_1
SummaryAlignment = SummaryAlignment_1

# Pruning the Beam
def beam_pruning(article, summaries, k, beam_size):
    # Calculate summary scores
    summary_scores = [SummaryAlignment(i , article) for i in summaries]
    # Sort indices of summary scores in descending order
    top_indexes = sorted(range(len(summary_scores)), key=lambda i: summary_scores[i], reverse=True)
    # Select the top summaries based on scores
    pre_prune_beam = np.array(summaries)[top_indexes]
    # Keep only the top 'beam_size' number of summaries
    post_prune_beam = pre_prune_beam[:beam_size]
    return post_prune_beam

# Expanding the Beam
def beam_expansion(article, summaries, k, beam_size, best_indexes):
    new_summaries = []
    for summary_index in range(len(summaries)):
        for sentence_index in best_indexes[:beam_size]:
            if sentence_index in summaries[summary_index]:
                # If sentence already in summary, do nothing
                pass
            else:
                # Otherwise, append new sentence to the summary
                new_summaries.append(list(summaries[summary_index]) + [sentence_index])
    return new_summaries

# the main beam search function
def beam_search(article, summaries, k, beam_size, best_indexes):
    if len(summaries[0]) == k:
        # If the current summaries are of length 'k', return the best summary in the beam
        return beam_pruning(article, summaries, k, 1)
    elif len(summaries) == beam_size:
        # If the number of summaries equals beam size, expand and prune the beam
        summaries = beam_expansion(article, summaries, k, beam_size, best_indexes)
        summaries = beam_pruning(article, summaries, k, beam_size)
        return beam_search(article, summaries, k, beam_size)       
    else:
        # Otherwise, prune, expand, and prune again before the next iteration
        summaries = beam_pruning(article, summaries, k, beam_size)
        summaries = beam_expansion(article, summaries, k, beam_size, best_indexes)
        summaries = beam_pruning(article, summaries, k, beam_size)
        return beam_search(article, summaries, k, beam_size)

# Decoding the scores to find the best summary order
def decode_scores(scores_in, article, hyperparameters):
    # Extract beam size and summary length 'k' from hyperparameters
    beam_size = hyperparameters['PureRougeSearch']['beam_size']
    k = hyperparameters['k']
    # Get the best initial best sentence indices based on input scores
    best_indexes = sorted(range(len(scores_in)), key=lambda i: scores_in[i], reverse=True)
    
    # Initialize with summaries of length 1
    initial_summaries = [[i] for i in best_indexes[:beam_size]]
    # Use beam search to find the best summary
    best_summary = beam_search(article, initial_summaries, k, beam_size, best_indexes)[0]
    # Calculate the ordering score of each permutation of the best summary
    ordering_scores = [SummaryAlignment_RL(summary, article) for summary in list(itertools.permutations(best_summary))]
    # Select the permutation with the highest score
    best_idxs = sorted(range(len(ordering_scores)), key=lambda i: ordering_scores[i], reverse=True)[0]
    best_summary = ordering_scores[best_idxs]

    # Score each sentence as 1 if it's in the best summary, else as 0
    sentence_scores = []
    for i in range(len(article)):
        if i in best_summary:
            sentence_scores.append(1)
        else:
            sentence_scores.append(0)
    return sentence_scores