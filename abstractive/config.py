#####################################################################################
'''                                                                                 
General Configurations             
'''
#####################################################################################

import torch
# Device for GPU speedup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# special tokens and their indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
# Minimum word frequency for inclusion in vocabulary
VOCAB_MIN_FREQ = 10