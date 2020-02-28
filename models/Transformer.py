import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
from models import Conv3D


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
                 num_classes = 500,
                 max_len = 15):
        super(CSL_Transformer,self).__init__()
        # build the feature extractor
        self.featureExtractor = Conv3D.resnet18(pretrained=True, sample_size=sample_size, 
                    sample_duration=clip_length, num_classes=num_classes)
        self.image_feature_dim = 500
        self.new_fc = nn.Linear(self.image_feature_dim, d_model)
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

    def extract_feature(self, images):
        # N x C x L x H x W, where N is set as 1
        #TODO support N > 1
        images = images.squeeze(0)
        size = images.size()
        # C x 16 x S x H x W, S is the sequence length
        images = images.view(size[0],self.clip_length,-1,size[2],size[3])
        # S x C x 16 x H x W
        images = images.permute(2,0,1,3,4)
        feature = self.featureExtractor(images)
        # S x D, D = d_model
        src = self.new_fc(feature)
        src = F.normalize(src,2)
        src = src.unsqueeze(1)
        src = src * math.sqrt(self.d_model)
        src = self.dropout(self.pos_encoder(src))
        return src


    def forward(self, images, tgt):       
        # Convert images to src sequence
        src = self.extract_feature(images)
        tgt = tgt.transpose(0,1)

        if self.tgt_subsequent_mask is None or self.tgt_subsequent_mask.size(0) != len(tgt):
            self.tgt_subsequent_mask = self.get_square_subsequent_mask(tgt)
        # if self.src_pad_mask is None or self.src_pad_mask.size(1) != len(src):
        #     self.src_pad_mask = self.get_pad_mask(src)
        # if self.tgt_pad_mask is None or self.tgt_pad_mask.size(1) != len(tgt):
        #     self.tgt_pad_mask = self.get_pad_mask(tgt)
        # if self.memory_pad_mask is None or self.memory_pad_mask.size(1) != len(src):
        #     self.memory_pad_mask = self.get_pad_mask(src)

        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.dropout(self.pos_encoder(tgt))
        out = self.transformer(src, tgt, 
                tgt_mask = self.tgt_subsequent_mask,
                src_key_padding_mask = self.src_pad_mask,
                tgt_key_padding_mask = self.tgt_pad_mask,
                memory_key_padding_mask = self.memory_pad_mask)
        out = self.out(out)
        return out

    def greedy_decode(self, images, max_len):
        # Convert images to src sequence
        src = self.extract_feature(images)
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
        one_hot = torch.zeros(ys.size()[0],self.vocab_size).cuda().scatter_(1,ys,1)
        return one_hot



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



