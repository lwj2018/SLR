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
        self.tgt_subsequent_mask = None

        self.embedding = nn.Embedding(src_vocab_size, d_model)
        self.pos_encoder = PostionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        self.transformer = nn.Transformer(d_model, nhead, num_enc_layers, 
                                          num_dec_layers, dim_feedforward, 
                                          dropout, activation)
        self.out = nn.Linear(d_model, tgt_vocab_size)

    def get_pad_mask(self, data):
        # the index of '<pad>' is 0
        # shape of mask is: N x T
        mask = data.eq(0).transpose(0,1)
        # mask = mask.masked_fill(mask == True, float('-inf')).masked_fill(mask == False, float(0.0))
        return mask

    def get_src_pad_mask(self, src, src_len_list):
        # shape of src is: S x N x E
        # shape fo src_len_list is: N
        # shape of mask is: N x S
        # mask = torch.zeros(src.size(1),src.size(0))
        mask = torch.ByteTensor(src.size(1),src.size(0))
        # mask.fill_(True)
        for i in range(mask.size(0)):
            for j in range(mask.size(1)):
                if j>=src_len_list[i]:
                    mask[i,j] = False
        mask = mask.cuda()
        return mask

    def get_square_subsequent_mask(self, tgt):
        seq_len = tgt.size(0)
        # see torch.triu return upper triangular part of the matrix except diagonal element
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal = 1)
        mask = mask.float().masked_fill(mask == 0, float(0.0)).masked_fill(mask == 1, float('-inf'))
        mask = mask.cuda()
        return mask

    def extract_image_feature(self, input):
        # shape of input is: N x S x C x 16 x H x W
        # After view, shape of input is: (NxS) x C x 16 x H x W
        N = input.size(0)
        input = input.view( (-1,) + input.size()[-4:] )
        feature = self.featureExtractor(input)
        # After feature extrace, shape of src is: (NxS) x E, E = d_model
        # src = self.new_fc(feature)
        src = F.normalize(feature,2)
        src = src.view(N,-1,src.size(-1))
        # After permute, shape of src is: S x N x E
        src = src.permute(1,0,2)
        src = src * math.sqrt(self.d_model)
        src = self.dropout(self.pos_encoder(src))
        return src

    def extract_skeleton_feature(self, input):
        # shape of input is: N x S x 16 x J x D, S is the sequence length
        # After view, shape of input is: (NxS) x 16 x J x D
        N = input.size(0)
        input = input.view( (-1,) + input.size()[-3:] )
        feature = self.featureExtractor(input)
        # After feature extrace, shape of src is: (NxS) x E, E = d_model
        # src = self.new_fc(feature)
        src = F.normalize(feature,2)
        src = src.view(N,-1,src.size(-1))
        # After permute, shape of src is: S x N x E
        src = src.permute(1,0,2)
        src = src * math.sqrt(self.d_model)
        src = self.dropout(self.pos_encoder(src))
        return src


    def forward(self, input, tgt, src_len_list, tgt_len_list):    
        # Convert input to src sequence
        if self.modal=='rgb':
            src = self.extract_image_feature(input)
        elif self.modal=='skeleton':
            src = self.extract_skeleton_feature(input)
        # After transpose, shape of tgt is: T x N
        tgt = tgt.transpose(0,1)

        # Masking
        if self.tgt_subsequent_mask is None or self.tgt_subsequent_mask.size(0)!=len(tgt):
            self.tgt_subsequent_mask = self.get_square_subsequent_mask(tgt)
        src_pad_mask = self.get_src_pad_mask(src,src_len_list)
        tgt_pad_mask = self.get_pad_mask(tgt)
        memory_pad_mask = self.get_src_pad_mask(src,src_len_list)

        # After embedding, shape of tgt is: T x N x E
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.dropout(self.pos_encoder(tgt))
        out = self.transformer(src, tgt, 
                tgt_mask = self.tgt_subsequent_mask,
                src_key_padding_mask = src_pad_mask,
                tgt_key_padding_mask = tgt_pad_mask,
                memory_key_padding_mask = memory_pad_mask)
        out = self.out(out)
        return out

    def greedy_decode(self, input, max_len):
        # Now support N > 1
        # Convert input to src sequence
        if self.modal=='rgb':
            src = self.extract_image_feature(input)
        elif self.modal=='skeleton':
            src = self.extract_skeleton_feature(input)
        N = src.size(1)
        model = self.transformer
        # shape of memory is: S x N x E
        memory = model.encoder.forward(src)
        # give the begin word
        ys = torch.ones(1, N, dtype=torch.long).fill_(1).cuda()
        # After embedding, shape of tgt is 1 x N x E
        tgt = self.embedding(ys) * math.sqrt(self.d_model)
        tgt = self.dropout(self.pos_encoder(tgt))
        for i in range(max_len-1):
            out = model.decoder.forward(tgt, memory, 
                            tgt_mask=self.get_square_subsequent_mask(tgt),
                            memory_mask=None)
            # shape of prob is: currT x N x vocab_size
            prob = self.out(out)
            # shape of next_word is: currT x N
            next_word = torch.argmax(prob, dim = 2)
            next_word = next_word.data[-1,:].unsqueeze(0)
            ys = torch.cat([ys,next_word], dim=0)
            # After embedding, shape of tgt is (currT+1) x N x E
            tgt = self.embedding(ys) * math.sqrt(self.d_model)
            tgt = self.dropout(self.pos_encoder(tgt))
        # shape of ys is: max_len x N
        # shape of prob is: max_len x N x vocab_size
        return prob

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



