import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import random
import pickle

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
        self.skeleton_dim = skeleton_dim
        self.rnn = nn.LSTM(input_size=self.skeleton_dim,
                          hidden_size=self.enc_hid_dim, bidirectional=True)
        self.fc1 = nn.Linear(2*enc_hid_dim,dec_hid_dim)
        self.fc2 = nn.Linear(2*enc_hid_dim,dec_hid_dim)

    def forward(self,x):
        # x: (T1 x N x (JxD))
        # outputs: (T1 x N x (2xenc_hid_dim))
        # hidden, cell: (1 x N x dec_hid_dim)
        outputs, (hidden,cell) = self.rnn(x)
        hidden = hidden.permute(1,0,2)
        hidden = hidden.flatten(start_dim=1)
        hidden = hidden.unsqueeze(0)
        cell = cell.permute(1,0,2)
        cell = cell.flatten(start_dim=1)
        cell = cell.unsqueeze(0)
        hidden = self.fc1(hidden)
        cell = self.fc2(cell)
        return outputs, (hidden,cell)

class Attention(nn.Module):
    def __init__(self,attn_dim=128,enc_hid_dim=e_dim,dec_hid_dim=d_dim):
        super(Attention,self).__init__()
        self.attn = nn.Linear(2*enc_hid_dim+dec_hid_dim,dec_hid_dim)

    def forward(self,encoder_outputs,decoder_hidden):
        # encoder_outputs: (T1 x N x (2xenc_hid_dim))
        # decoder_hidden: (1 x N x dec_hid_dim)
        # attention: (T1 x N)
        src_len = encoder_outputs.size(0)
        decoder_hidden_rep = decoder_hidden.repeat(src_len,1,1)
        energy = self.attn(torch.cat([encoder_outputs,decoder_hidden_rep],dim=2))
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
        # hidden(1 x N x dec_hid_dim): last hidden state
        # cell(1 x N x dec_hid_dim): last cell state
        # encoder_outputs(T1 x N x (2xenc_hid_dim))
        # a: (T1 x N)
        # calculate context vector
        a = self.attention(encoder_outputs,hidden)
        context = self.weighted_sum(a,encoder_outputs)
        context = context.permute(1,0,2)
        # decode one step
        decoder_input = torch.cat([input,velocity,context],dim=2)
        outputs, (hidden, cell) = self.rnn(decoder_input, (hidden,cell))
        outputs = self.fc(outputs)
        velocity = outputs - input
        return outputs, velocity, (hidden, cell)


class Text2Sign(nn.Module):
    def __init__(self, encoder, decoder, word2gloss_database):
        super(Text2Sign, self).__init__()
        self.db = word2gloss_database
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, text, target, teacher_forcing_ratio=0.5):
        # text: (N x T)
        # target: (N x T2 x J x D)
        # only support N = 1 now
        text = text.squeeze(0)
        trg_len = target.size()[1]
        # target: (1 x N x T2 x (JxD))
        target = target.flatten(start_dim=2)
        target = target.unsqueeze(0) # add the fake "temporal" dim

        # transform input text to skeleton sequence of source
        # src: (T1 x J x D)
        src = []
        for word in text:
            gloss = torch.Tensor(self.db[word.item()])
            src.append(gloss)
        src = torch.cat(src,dim=0)
        src = src.cuda()

        # Encode source sequence
        # src: (T1 x N x (JxD))
        src = src.flatten(start_dim=1)
        src = src.unsqueeze(1)
        encoder_outputs, (hidden,cell) = self.encoder(src)

        # first input
        # velocity is zero
        input = target[:,:,0,:]
        velocity = torch.zeros(input.size()).cuda()

        outputs = []

        for t in range(1, trg_len):
            # decode
            output, velocity, (hidden, cell) = self.decoder(input, velocity, hidden, cell, encoder_outputs)

            # store prediction
            outputs.append(output)

            # decide whether to do teacher foring
            teacher_force = random.random() < teacher_forcing_ratio

            # apply teacher forcing
            gt = target[:,:,t,:]
            velocity = gt-input if teacher_force else velocity
            input    = gt if teacher_force else output

        # outputs: (T2 x N x (JxD))
        outputs = torch.cat(outputs,dim=0)
        outputs = outputs.reshape(-1,outputs.size(1),joint,dimension)
        outputs = outputs.transpose(0,1)
        return outputs


# Test
if __name__ == '__main__':
    # test encoder
    encoder = Encoder()
    # x = torch.randn(200,1,joint*dimension)
    # outputs, (hidden, cell) = encoder(x)
    # print(f"{outputs} {hidden} {cell}")

    # test decoder
    decoder = Decoder()
    # input = torch.randn(1,joint*dimension)
    # velocity = torch.randn(1,joint*dimension)
    # hidden = torch.randn(1,1,2*e_dim)
    # cell = torch.randn(1,1,2*e_dim)
    # encoder_outputs = torch.randn(200,1,2*e_dim)
    # outputs, velocity, (hidden,cell) = decoder(input,velocity,hidden,cell,encoder_outputs)

    # # test seq2seq
    db_file = open('/home/liweijie/projects/SLR/obj/word2gloss.pkl','rb')
    database = pickle.load(db_file)
    text2sign = Text2Sign(encoder=encoder, decoder=decoder, word2gloss_database=database)
    text = torch.LongTensor(1, 16).random_(0, 500)
    target = torch.randn(1,300,joint,dimension)
    outputs = text2sign(text,target)
    print(outputs.size())
