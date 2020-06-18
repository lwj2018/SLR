import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import random

"""
Implementation of Text to Sign Model
Decoder: step-by-step decode based on attention from the input skeleton
"""
joint = 21*2+25
dimension = 2
e_dim = 128
d_dim = 256

class Encoder(nn.Module):
    def __init__(self,enc_hid_dim=e_dim,dec_hid_dim=d_dim,
            skeleton_dim=joint*dimension):
        super(Encoder, self).__init__()
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.rnn = nn.LSTM(input_size=self.skeleton_dim,
                          hidden_size=self.enc_hid_dim, bidirectional=True)
        self.fc1 = nn.Linear(2*enc_hid_dim,dec_hid_dim)
        self.fc2 = nn.Linear(2*enc_hid_dim,dec_hid_dim)

    def forward(self,x):
        # x: (T1 x N x (JxD))
        # outputs: (T1 x N x (2xenc_hid_dim))
        # hidden, cell: (N x dec_hid_dim)
        outputs, (hidden,cell) = self.rnn(x)
        hidden = self.fc1(hidden)
        cell = self.fc2(cell)
        return outputs, (hidden,cell)

class Attention(nn.Module):
    def __init__(self,attn_dim=128,enc_hid_dim=e_dim,dec_hid_dim=d_dim):
        super(Attention,self).__init__()
        self.attn = nn.Linear(2*enc_hid_dim+dec_hid_dim,dec_hid_dim)

    def forward(self,encoder_outputs,decoder_hidden):
        # encoder_outputs: (T1 x N x (2xenc_hid_dim))
        # decoder_hidden: (N x dec_hid_dim)
        # attention: (T1 x N)
        src_len = encoder_outputs.size(0)
        decoder_hidden_rep = decoder_hidden.unsqueeze(0).repeat(src_len,1,1)
        energy = self.attn(torch.cat([encoder_outputs,decoder_hidden_rep]))
        attention = torch.sum(energy,dim=2)
        return F.softmax(attention,dim=0)

class Decoder(nn.Module):
    def __init__(self,enc_hid_dim=e_dim,dec_hid_dim=d_dim,
            dropout=0.1, skeleton_dim=joint*dimension):
        super(Decoder, self).__init__()
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.rnn = nn.LSTM(2*skeleton_dim+2*enc_hid_dim, dec_hid_dim)
        self.attention = Attention()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(dec_hid_dim, skeleton_dim)

    def weighted_sum(self,attention,encoder_outputs):
        # context: (N x 1 x (2xenc_hid_dim))
        attention = attention.permute(1,0)
        attention = attention.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1,0,2)
        context = torch.bmm(attention,encoder_outputs)
        return context

    def forward(self, input, velocity, hidden, cell, encoder_outputs):
        # input(N x (JxD)): last skeleton
        # velocity(N x (JxD)): last velocity
        # hidden(N x dec_hid_dim): last hidden state
        # cell(N x dec_hid_dim): last cell state
        # encoder_outputs(T1 x N x (2xenc_hid_dim))
        input = input.unsqueeze(0)   # add the fake "temporal" dim
        velocity = velocity.unsqueeze(0)   # add the fake "temporal" dim
        # a: (T1 x N)
        # calculate context vector
        a = self.attn(encoder_outputs,hidden)
        context = self.weighted_sum(attention,encoder_outputs)
        


class Text2Sign(nn.Module):
    def __init__(self, encoder, decoder, device, word2gloss_database):
        super(Text2Sign, self).__init__()
        self.db = word2gloss_database
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, text, target, teacher_forcing_ratio=0.5):
        # text: (N x T)
        # target: (N x T2 x J x D)
        # only support N = 1 now
        text = text.squeeze(0)
        target = target.squeeze(0)
        trg_len = target.size()[0]

        # transform input text to skeleton sequence of source
        # src: (T1 x J x D)
        src = []
        for word in range(text):
            gloss = self.db[word]
            src.append(gloss)
        src = torch.cat(src,dim=0)

        # Encode source sequence
        # src: (T1 x N x (JxD))
        src = src.flatten(start_dim=1)
        src = src.unsqueeze(1)
        encoder_outputs, (hidden,cell) = self.encoder(src)

        # first input
        # velocity is zero
        input = target[0]
        input = input.unsqueeze(0) # add the fake "batch" dim
        input = input.flatten(start_dim=1)
        velocity = torch.zeros(input.size())
        velocity = velocity.unsqueeze(1) # add the fake "batch" dim

        for t in range(1, trg_len):
            # decode
            output, velocity, (hidden, cell) = self.decoder(input, velocity, hidden, cell, encoder_outputs)

            # store prediction
            outputs[t] = output

            # decide whether to do teacher foring
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token
            top1 = output.argmax(1)

            # apply teacher forcing
            input = target[:,t] if teacher_force else top1

        return outputs


# Test
if __name__ == '__main__':
    # test encoder
    encoder = Encoder(lstm_hidden_size=512)
    # imgs = torch.randn(16, 3, 8, 128, 128)
    # print(encoder(imgs))

    # test encoderPlus
    encoderPlus = EncoderPlus(lstm_hidden_size=512)
    # imgs = torch.randn(16, 3, 8, 128, 128)
    # print(encoderPlus(imgs))

    # test decoder
    decoder = Decoder(output_dim=500, emb_dim=256, enc_hid_dim=512, dec_hid_dim=512, dropout=0.5)
    # input = torch.LongTensor(16).random_(0, 500)
    # hidden = torch.randn(16, 512)
    # cell = torch.randn(16, 512)
    # context = torch.randn(16, 512)
    # print(decoder(input, hidden, cell, context))

    # test seq2seq
    device = torch.device("cpu")
    # seq2seq = Seq2Seq(encoder=encoder, decoder=decoder, device=device)
    seq2seq = Seq2Seq(encoder=encoderPlus, decoder=decoder, device=device)
    imgs = torch.randn(16, 3, 8, 128, 128)
    target = torch.LongTensor(16, 8).random_(0, 500)
    print(seq2seq(imgs, target).argmax(dim=2).permute(1,0)) # batch first
