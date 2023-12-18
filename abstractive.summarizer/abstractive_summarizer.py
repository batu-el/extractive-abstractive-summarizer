#####################################################################################
'''                                                                                 
# Abstractive Summarizer
# References:   https://github.com/MichSchli/L90-Summarization
                https://pytorch.org/tutorials/beginner/transformer_tutorial.html
                https://pytorch.org/tutorials/beginner/translation_transformer.html
                https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
#               
'''
#####################################################################################

# INITIALIZATION
# for initializing the model parameters in training
from torch.nn.init import xavier_normal # alternative initializations can be found at torch.nn.init
# CONFIGURATIONS
import config
# UTILITIES
from utils.dataset import CNNDMdataset
from utils.masks import create_mask
from utils.transforms import construct_token_transform, construct_vocab_transform, construct_tensor_transform, construct_text_transform
# DECODERS
from decoders.greedy import greedy_decode
from decoders.beamsearch import beam_search_decode
from decoders.temperature import temperature_decode
from decoders.mbrd import mbr_decode, mbr_decode_verbose, candidates_set
from decoders.uniform import uniform_decode
# TRANSFORMER MODEL
from EncoderDecoderTransformer import EncoderDecoderTransformer
# EVALUATION
from evaluation.rouge_evaluator import RougeEvaluator
# PYTORCH
import torch
# TRACKING PROGRESS
from timeit import default_timer as timer
import tqdm
import time
# DATA MANAGEMENT
import numpy as np
import pandas as pd

class AbstractiveSummarizer:
    def __init__(self):
        # transforms
        self.token_transform = None # to transform the stings into tokens
        self.vocab_transform = None # to transform the tokens into indexes in the vocabulary
        self.tensor_transform = None # to transform vocabulary indexes into tensors
        self.text_transform = None # to apply these transformations sequentially

        # assume a common vocabulary for source and target
        self.VOCAB_SIZE = None #len(self.vocab_transform) - different from Vaswani et al. (2017), who used different vocabularies for source and target languages
        self.EMB_SIZE = None # size of token embeddings and positional encodings (512 in Vaswani et al. (2017))
        self.NHEAD = None # number of attention heads in all attention sublayers (8 in Vaswani et al. (2017))
        self.FFN_HID_DIM = None # number of dimensions in the hidden layer of feed-forward sublayer (2048 in Vaswani et al. (2017))
        self.NUM_ENCODER_LAYERS = None # number of encoder layers (6 in Vaswani et al. (2017))
        self.NUM_DECODER_LAYERS = None # number of decoder layers (6 in Vaswani et al. (2017))

        self.BATCH_SIZE = None

        self.model = None
        self.loss_fn = None
        self.optimizer = None
        self.NUM_EPOCHS = None

        self.train_losses = None
        self.val_losses = None
        self.model_paths = None
        self.model_state = None

        self.model_id = '-'.join(time.ctime(time.time()).replace(':', ' ').split(' ')[2:5])

    def collate_fn(self , batch):
        # lists to hold processed source and target
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            # convert to tensor
            src_batch.append(self.text_transform(src_sample.rstrip("\n")))
            tgt_batch.append(self.text_transform(tgt_sample.rstrip("\n")))
        # pad the sequences to ensure they have the same length
        src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=config.PAD_IDX)
        tgt_batch = torch.nn.utils.rnn.pad_sequence(tgt_batch, padding_value=config.PAD_IDX)
        return src_batch, tgt_batch
    
    def train_epoch_and_compute_loss(self, train_iter):
        # put the model in training mode
        self.model.train()
        losses = 0
        # use a dataloader to organize, shuffle, and iterate over data
        train_dataloader = torch.utils.data.DataLoader(train_iter, batch_size=self.BATCH_SIZE, collate_fn=self.collate_fn)
        # zero optimizer
        self.optimizer.zero_grad()
        for batch_idx , (src, tgt) in tqdm.tqdm(enumerate(train_dataloader) , desc='Progress within Training Epoch'):
            # src and tgt sequences to device
            src = src.to(config.DEVICE)
            tgt = tgt.to(config.DEVICE)
            # shift the decoder input to the left
            tgt_input = tgt[:-1, :]
            # create masks
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
            # forward pass
            logits = self.model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
            # shift the output to the right for calculating the loss
            tgt_out = tgt[1:, :]
            # compute loss 
            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            # accumulate the loss.
            losses += loss.item()
            # normalize the loss
            loss = loss / self.GRAD_ACC
            # compute grad
            loss.backward()
            # update model parameters every GRAD_ACC batches or at the end of the data
            if ((batch_idx % self.GRAD_ACC) == 0) or (batch_idx + 1 == len(train_dataloader)):
                self.optimizer.step()
                self.optimizer.zero_grad()            
        return losses / len(list(train_dataloader))

    def compute_validation_score(self, val_iter):
        # put the model in evaluation mode
        self.model.eval()
        losses = 0
        # data loader to iterate over data
        val_dataloader = torch.utils.data.DataLoader(val_iter, batch_size=self.BATCH_SIZE, collate_fn=self.collate_fn)
        for src, tgt in tqdm.tqdm(val_dataloader , desc='Progress in Validation'):
            # src and tgt sequences to device
            src = src.to(config.DEVICE)
            tgt = tgt.to(config.DEVICE)
            # shift the decoder input to the left
            tgt_input = tgt[:-1, :]
            # create masks
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
            # forward pass
            logits = self.model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
            # shift the output to the right for calculating the loss
            tgt_out = tgt[1:, :]
            # compute loss
            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            losses += loss.item()
        return losses / len(list(val_dataloader))

    def train(self, X, y, val_X, val_y, 
              NUM_ENCODER_LAYERS = 1, 
              NUM_DECODER_LAYERS = 1, 
              EMB_SIZE = 64, 
              NHEAD = 1 ,
              FFN_HID_DIM = 32 ,
              BATCH_SIZE = 8,
              GRAD_ACC = 1,
              LEARNING_RATE = 0.0001,
              NUM_EPOCHS = 2
              ):
        train_iter = CNNDMdataset(X , y)
        val_iter = CNNDMdataset(val_X , val_y)

        # preprocessing utilities
        self.token_transform = construct_token_transform()
        self.vocab_transform = construct_vocab_transform(train_iter)
        self.tensor_transform = construct_tensor_transform()
        self.text_transform = construct_text_transform(self.token_transform , self.vocab_transform, self.tensor_transform)
        self.VOCAB_SIZE = len(self.vocab_transform)
        # model specifications
        self.NUM_ENCODER_LAYERS = NUM_ENCODER_LAYERS
        self.NUM_DECODER_LAYERS = NUM_DECODER_LAYERS
        self.EMB_SIZE = EMB_SIZE
        self.NHEAD = NHEAD
        self.FFN_HID_DIM = FFN_HID_DIM
        # training specifications
        self.BATCH_SIZE = BATCH_SIZE
        self.GRAD_ACC = GRAD_ACC
        self.LEARNING_RATE = LEARNING_RATE
        # initialize the model parameters using Glorot Normal
        self.model = EncoderDecoderTransformer(self.NUM_ENCODER_LAYERS, self.NUM_DECODER_LAYERS, self.EMB_SIZE, self.NHEAD, self.VOCAB_SIZE, self.FFN_HID_DIM)
        for p in self.model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal(p)
        # put to device
        self.model = self.model.to(config.DEVICE)
        # cross entropy loss
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=config.PAD_IDX)
        # adam optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)
        # lists to save losses
        self.train_losses = []
        self.val_losses = []
        self.model_paths = []
        # iterate over epochs
        NUM_EPOCHS = NUM_EPOCHS
        for epoch in range(1, NUM_EPOCHS+1):
            # measure the time of each epoch
            start_time = timer()
            # train and compute loss
            train_loss = self.train_epoch_and_compute_loss(train_iter)
            end_time = timer()
            # compute validation score
            val_loss = self.compute_validation_score(val_iter)
            # save losses 
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            # add the current path to model paths
            self.model_paths.append(f'{self.model_id}-epoch-{epoch}-trainloss-{train_loss:.3f}-valloss-{val_loss:.3f}.pt')
            # save model
            torch.save(self.model.state_dict(), 'model_states/'+self.model_paths[-1])
            # display the progress
            print((f"Epoch: {epoch}, Training loss: {train_loss:.3f}, Validation loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
        # get the best model and load it
        best_model_index = np.argmin(self.val_losses)
        self.model_state = self.model_paths[best_model_index]
        self.model.load_state_dict(torch.load( 'model_states/'+self.model_state))
        return 'Done'
    
    def generate(self, decoder, src_sentence):
        # set the model to evaluation mode
        self.model.eval()
        # transform to tensor
        src = self.text_transform(src_sentence).view(-1, 1)
        # number of tokens in the source sentence
        num_tokens = src.shape[0]
        # create source mask to prevent unnecessary attention for padding tokens  
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        try:
            # for all decoding methods except MBR
            tgt_tokens = decoder( self ,  src, src_mask, num_tokens, config.BOS_IDX).flatten()
            return " ".join(self.vocab_transform.lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")
        except:
            # for Minimum Bayes Risk Decoding
            return decoder( self ,  src, src_mask, num_tokens, config.BOS_IDX)

    def predict( self, X, decoder=greedy_decode):
        for article in tqdm.tqdm(X, desc="Running abstractive summarizer"):
            # iterate and summarize articles
            yield self.generate(decoder, article)