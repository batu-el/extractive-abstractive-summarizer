import config
import torch
    
''' TOKEN TRANSFORM '''
from torchtext.data.utils import get_tokenizer
def construct_token_transform():
    return get_tokenizer('spacy', language='en_core_web_sm') #get_tokenizer('basic_english')

''' VOCAB TRANSFORM '''
from torchtext.vocab import build_vocab_from_iterator
# Helper function to consturct a single vocabulary for source and target
def yield_tokens(data_iter, token_transform):
    pairs = ['Article' , "Summary"]
    for i in range(len(pairs)):
        for data_sample in data_iter:
            # apply token transform to data sample
            yield token_transform(data_sample[i])

def construct_vocab_transform(train_iter):
    # define the token transform
    token_transform = construct_token_transform()
    # create torchtext vocab object
    vocab_transform = build_vocab_from_iterator(yield_tokens(train_iter, token_transform), min_freq= config.VOCAB_MIN_FREQ, specials=config.special_symbols, special_first=True)
    # set ``UNK_IDX`` as  default index
    vocab_transform.set_default_index(config.UNK_IDX)
    return vocab_transform

''' TENSOR TRANSFORM '''

def tensor_transform(token_ids):
    # add BOS/EOS and create tensor for input sequence indices
    return torch.cat((torch.tensor([config.BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([config.EOS_IDX])))
def construct_tensor_transform():
    # wrapper function
    return tensor_transform

''' TEXT TRANSFORM '''
def sequential_transforms(*transforms):
    # sequential application of functions
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

def construct_text_transform(token_transform, vocab_transform, tensor_transform):
    # sequentially apply transforms
    # tokenize + numericalize + add BOS/EOS + convert to tensor
    text_transform = sequential_transforms(token_transform, # tokenization
                                            vocab_transform, # numericalization
                                            tensor_transform) # ddd BOS/EOS and create tensor
    return text_transform
