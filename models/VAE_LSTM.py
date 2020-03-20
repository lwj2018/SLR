import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
from models import VAE, HCN
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence

"""
Implementation of Transformer
"""
class vae_lstm(nn.Module):
    def __init__(self,  
                 vocab_size,
                 dim_feedforward = 2048, 
                 dropout = 0.1, 
                 activation = 'relu',
                 clip_length = 32,
                 num_classes = 500,
                 hidden_dim = 512):
        super(vae_lstm,self).__init__() 
        # build the feature extractor & vae
        self.featureExtractor = HCN.hcn(num_classes,length=clip_length)
        self.vae = VAE.VAE(num_classes) 
        self.clip_length = clip_length

        self.lstm = nn.LSTM(input_size=num_classes,hidden_size=hidden_dim,num_layers=3,bidirectional=True)
        
        self.out = nn.Linear(2*hidden_dim, vocab_size)

    def extract_skeleton_feature(self, input, N):
        # shape of input is: (NxS) x 16 x J x D, S is the sequence length
        feature = self.featureExtractor(input)
        # After feature extract, shape of src is: (NxS) x num_class
        src = F.normalize(feature,2)
        src = src.view(N,-1,src.size(-1))
        # After permute, shape of src is: S x N x num_class
        src = src.permute(1,0,2)
        return src


    def forward(self, input, src_len_list):  
        N =  input.size(0)
        input = input.view( (-1,) + input.size()[-3:] )
        # resume skeleton
        input = self.vae(input)[0]
        # normalize along the J dimension
        input = F.normalize(input,1,dim=2)
        # Convert input to src sequence
        src = self.extract_skeleton_feature(input,N)
        # LSTM forward
        pack = pack_padded_sequence(src,src_len_list)
        out,_ = self.lstm(pack)
        out,_ = pad_packed_sequence(out)
        # final fc
        out = self.out(out)
        # log softmax
        out = F.log_softmax(out,2)
        return out

    # def train(self, mode=True):
    #     """
    #     Override the default train() to freeze the VAE parameters
    #     :return:
    #     """
    #     for param in self.vae.parameters():
    #         param.requires_grad = False

 



