import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
from models import Conv3D, HCN


"""
Implementation of Transformer
"""
class CSL_Transformer(nn.Module):
    def __init__(self,  
                 src_vocab_size,
                 tgt_vocab_size,
                 d_model = 512, 
                 nhead = 8, num_enc_layers = 6, 
                 num_dec_layers = 6, 
                 dim_feedforward = 2048, 
                 dropout = 0.1, 
                 activation = 'relu',
                 clip_length = 16,
                 sample_size = 128,
                 num_classes = 512,
                 max_len = 15,
                 modal = 'skeleton'):
        super(CSL_Transformer,self).__init__()
        # build the feature extractor
        self.modal = modal
        if modal=='rgb':
            self.featureExtractor = Conv3D.resnet18(pretrained=True, sample_size=sample_size, 
                        sample_duration=clip_length, num_classes=num_classes)
        else:
            self.featureExtractor = HCN.hcn(num_classes,length=clip_length)
        self.feature_dim = num_classes
        # self.new_fc = nn.Linear(self.feature_dim, d_model)
        self.clip_length = clip_length
        self.max_len = max_len
        self.vocab_size = src_vocab_size

        self.d_model = d_model
        self.src_pad_mask = None
        self.tgt_pad_mask = None
        self.memory_pad_mask = None
        self.tgt_subsequent_mask = None

        self.embedding = nn.Embedding(src_vocab_size, d_model)
        self.pos_encoder = PostionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        self.transformer = nn.Transformer(d_model, nhead, num_enc_layers, 
                                          num_dec_layers, dim_feedforward, 
                                          dropout, activation)
        self.out = nn.Linear(d_model, tgt_vocab_size)

    def get_pad_mask(self, data):
        # the index of '<pad>' is 1
        mask = data.eq(1).transpose(0,1)
        mask = mask.masked_fill(mask == True, float('-inf')).masked_fill(mask == False, float(0.0))
        return mask


    def get_square_subsequent_mask(self, tgt):
        seq_len = tgt.size(0)
        # see torch.triu return upper triangular part of the matrix except diagonal element
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal = 1)
        mask = mask.float().masked_fill(mask == 0, float(0.0)).masked_fill(mask == 1, float('-inf'))
        mask = mask.cuda()
        return mask

    def extract_image_feature(self, input):
        # N x C x L x H x W, where N is set as 1
        #TODO support N > 1
        input = input.squeeze(0)
        size = input.size()
        # C x S x 16 x H x W, S is the sequence length
        input = input.view(size[0],-1,self.clip_length,size[-2],size[-1])
        # S x C x 16 x H x W
        input = input.permute(1,0,2,3,4)
        feature = self.featureExtractor(input)
        # S x D, D = d_model
        # src = self.new_fc(feature)
        src = F.normalize(feature,2)
        src = src.unsqueeze(1)
        src = src * math.sqrt(self.d_model)
        src = self.dropout(self.pos_encoder(src))
        return src

    def extract_skeleton_feature(self, input):
        # N x T x J x D, where N is set as 1
        #TODO support N > 1
        input = input.squeeze(0)
        size = input.size()
        # S x 16 x J x D, S is the sequence length
        input = input.view(-1,self.clip_length,size[-2],size[-1])
        # S x 16 x J x D
        feature = self.featureExtractor(input)
        # S x D, D = d_model
        # src = self.new_fc(feature)
        src = F.normalize(feature,2)
        src = src.unsqueeze(1)
        src = src * math.sqrt(self.d_model)
        src = self.dropout(self.pos_encoder(src))
        return src


    # def forward(self, input, tgt):       
    #     # Convert input to src sequence
    #     if self.modal=='rgb':
    #         src = self.extract_image_feature(input)
    #     elif self.modal=='skeleton':
    #         src = self.extract_skeleton_feature(input)
    #     tgt = tgt.transpose(0,1)

    #     if self.tgt_subsequent_mask is None or self.tgt_subsequent_mask.size(0) != len(tgt):
    #         self.tgt_subsequent_mask = self.get_square_subsequent_mask(tgt)
    #     # if self.src_pad_mask is None or self.src_pad_mask.size(1) != len(src):
    #     #     self.src_pad_mask = self.get_pad_mask(src)
    #     # if self.tgt_pad_mask is None or self.tgt_pad_mask.size(1) != len(tgt):
    #     #     self.tgt_pad_mask = self.get_pad_mask(tgt)
    #     # if self.memory_pad_mask is None or self.memory_pad_mask.size(1) != len(src):
    #     #     self.memory_pad_mask = self.get_pad_mask(src)

    #     tgt = self.embedding(tgt) * math.sqrt(self.d_model)
    #     tgt = self.dropout(self.pos_encoder(tgt))
    #     out = self.transformer(src, tgt, 
    #             tgt_mask = self.tgt_subsequent_mask,
    #             src_key_padding_mask = self.src_pad_mask,
    #             tgt_key_padding_mask = self.tgt_pad_mask,
    #             memory_key_padding_mask = self.memory_pad_mask)
    #     out = self.out(out)
    #     return out

    def greedy_decode(self, input, max_len):
        # Convert input to src sequence
        if self.modal=='rgb':
            src = self.extract_image_feature(input)
        elif self.modal=='skeleton':
            src = self.extract_skeleton_feature(input)
        model = self.transformer
        memory = model.encoder.forward(src)
        # give the begin word
        ys = torch.ones(1, 1, dtype=torch.long).fill_(0).cuda()
        ye = torch.ones(1, 1, dtype=torch.long).fill_(1).cuda()
        tgt = self.embedding(ys) * math.sqrt(self.d_model)
        tgt = self.dropout(self.pos_encoder(tgt))
        for i in range(max_len-1):
            out = model.decoder.forward(tgt, memory, 
                            tgt_mask=self.get_square_subsequent_mask(tgt),
                            memory_mask=None)
            prob = self.out(out)
            next_word = torch.argmax(prob, dim = 2)
            next_word = next_word.data[-1]
            ys = torch.cat([ys,
                            torch.ones(1, 1, dtype=torch.long).fill_(next_word.squeeze()).cuda()], dim=0)
            if next_word == 1:
                break
            tgt = self.embedding(ys) * math.sqrt(self.d_model)
            tgt = self.dropout(self.pos_encoder(tgt))
        # S x E, S is sequence length
        one_hot = torch.zeros(ys.size()[0],self.vocab_size).cuda().scatter_(1,ys,1)
        return one_hot

    def forward(self,input):
        # Convert input to src sequence
        if self.modal=='rgb':
            src = self.extract_image_feature(input)
        elif self.modal=='skeleton':
            src = self.extract_skeleton_feature(input)
        model = self.transformer
        memory = model.encoder.forward(src)
        # shape of out is: T x N x vocab_size
        out = self.out(memory)
        return out


class PostionalEncoding(nn.Module):
    """docstring for PostionEncoder"""
    def __init__(self, d_model, max_len = 5000):
        super(PostionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp( - torch.arange(0, d_model, 2).float() * math.log(10000) / d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0,1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x



