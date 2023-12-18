#####################################################################################
'''                                                                                 
# Sequence to Sequence Transformer 
# based on the paper "Attention Is All You Need". Vaswani et al. (2017)
# References:   https://arxiv.org/abs/1706.03762
                https://pytorch.org/tutorials/beginner/transformer_tutorial.html
                https://pytorch.org/tutorials/beginner/translation_transformer.html
                https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
#               
'''
#####################################################################################

from torch import Tensor
import torch
import torch.nn as nn
import math

from Transformer import Transformer

# Token Embeddings:
# from the paper "Attention Is All You Need". Vaswani et al. (2017)
# I use this class to convert tensor of input indices into corresponding tensor of token embeddings
# Reference: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size
    def forward(self, tokens: Tensor):
        # convert the input tokens into long data type to pass in embedding layer
        # Following Vaswani et al. (2017), multiply the output of the embedding layer by the square root of the embedding size
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Positional Encodings:
# from the paper "Attention Is All You Need". Vaswani et al. (2017)
# I use this class to add positional encoding to the token embedding.
# This introduces a notion of word order, which transformer architecture doesn't have otherwise
# Reference: https://pytorch.org/tutorials/beginner/translation_transformer.html
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout, maxlen= 5000):
        super(PositionalEncoding, self).__init__()
        # denominator for the positional encoding formula, exponential decay based on the embedding size.
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        # positional array with maximum length
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        # initialize tensor with zeros
        pos_embedding = torch.zeros((maxlen, emb_size))
        # apply sine to even indices as in Vaswani et al. (2017)
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        # apply cosine to odd indices as in Vaswani et al. (2017)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        # add extra dimension
        pos_embedding = pos_embedding.unsqueeze(-2)
        # initialize dropout layer
        self.dropout = nn.Dropout(dropout)
        # register 'pos_embedding' as a buffer, not a model parameter.
        self.register_buffer('pos_embedding', pos_embedding)
    def forward(self, token_embedding: Tensor):
        # add the token embeddings and the positional embeddings and apply dropout following Vaswani et al. (2017)
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# Encoder-Decoder Transformer:
# from the paper "Attention Is All You Need". Vaswani et al. (2017)
# Unlike the implementations for Nueral Machine Translation (NMT) that assume different vocabularies for source and target
# My implementation assumes a common vocabulary for souce and target, which is reasonable in the case of Abstractive Summarization
class EncoderDecoderTransformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, emb_size, nhead, vocab_size, dim_feedforward = 512, dropout = 0.1):
        super(EncoderDecoderTransformer, self).__init__()
        # initialize the transformer model
        self.transformer = Transformer(d_model=emb_size, nhead=nhead,num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,dim_feedforward=dim_feedforward,dropout=dropout)
        # linear layer maps the transformer output back to the vocabulary
        # this is then used to calculate a probability distribution over the vocabulary during inference
        self.generator = nn.Linear(emb_size, vocab_size)
        # token embedding layer for source and target
        self.tok_emb = TokenEmbedding(vocab_size, emb_size)
        # positional encodings
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self,src ,trg ,src_mask,tgt_mask,src_padding_mask,tgt_padding_mask,memory_key_padding_mask):
        # pass source through token embedding then positional encoding
        src_emb = self.positional_encoding(self.tok_emb(src))
        # pass target through token embedding then positional encoding
        tgt_emb = self.positional_encoding(self.tok_emb(trg))
        # pass the embedded source and target through the transformer
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None, src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        # pass output through linear layer to map it back to the vocab
        final_outs = self.generator(outs)
        return final_outs
    
    def encode(self, src: Tensor, src_mask: Tensor):
        # encode the source input using the encoder from transformer
        return self.transformer.encoder(self.positional_encoding(self.tok_emb(src)), src_mask)
    
    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        # decode the target input using the decoder from transformer
        return self.transformer.decoder(self.positional_encoding(self.tok_emb(tgt)), memory, tgt_mask)



