import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self,num_class, in_channel=2,
                            length=32,
                            num_joint=4+2*21+70,
                            dropout=0.2,
                            latent_dim=256):
        super(VAE, self).__init__()
        self.encoder = Encoder(num_class,in_channel,length,num_joint,dropout)
        self.decoder = Decoder(num_class,in_channel,length,num_joint,dropout)
        self.classifier = nn.Linear(?,num_class)
        self.fc_mu = nn.Linear(num_class,latent_dim)
        self.fc_var = nn.Linear(num_class,latent_dim)

    def encode(self,input):
        out = self.encoder(input)
        mu = self.fc_mu(out)
        log_var = self.fc_var(out)
        return [mu,log_var]

    def decode(self,input):
        out = self.decoder(input)
        return out

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self,x):
        z = self.encoder(x)
        return z


class Encoder(nn.Module):
    def __init__(self,num_class,in_channel,length,num_joint,dropout):
        super(Encoder,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel,64,1,1,padding=0),
            nn.ReLU()
            )
        self.conv2 = nn.Conv2d(64,32,(3,1),1,padding=(1,0))
        self.conv3 = nn.Conv2d(num_joint,32,3,2,padding=1)     
        self.conv4 = nn.Sequential(
            nn.Conv2d(32,64,3,2,padding=1),
            nn.Dropout2d(p=dropout),
        )
                
        self.conv5 = nn.Sequential(
            nn.Conv2d(64,128,3,2,padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=dropout),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128,256,3,2,padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=dropout),
        )

        # scale related to total number of maxpool layer
        scale = 16
        self.fc7 = nn.Sequential(
            nn.Linear(256*(length//scale)*(32//scale),256),
            nn.ReLU(),
            nn.Dropout2d(p=dropout)
        )
        self.fc8 = nn.Linear(256,num_class)

    def forward(self,input):
        # shape of input is: N x T x J x D
        input = self.select_indices(input)
        input = input.permute(0,3,1,2)

        # After permute, shape of input is N x D x T x J
        # learn point level feature
        out = self.conv1(input)
        out = self.conv2(out)

        # After permute, shape of out is N x J x T x D
        # learn joint co-occurance
        out = out.permute(0,3,2,1).contiguous()        
        out = self.conv3(out)
        out = self.conv4(out)

        out = self.conv5(out)
        out = self.conv6(out)

        # After conv, shape of out is:  N x J x S(T/16) x D
        out = out.view(out.size(0),-1)
        print(out.size())
        out = self.fc7(out)
        out = self.fc8(out)

        t = out
        assert not ((t != t).any())# find out nan in tensor
        assert not (t.abs().sum() == 0) # find out 0 tensor
        # N x C (num_class)
        return out

    def select_indices(self,input):
        left_arm = input[:,:,[3,4],:]
        right_arm = input[:,:,[6,7],:]
        left_hand = input[:,:,95:116,:]
        right_hand = input[:,:,116:137,:]
        face = input[:,:,25:95,:]
        x = torch.cat([left_arm,right_arm,left_hand,\
            right_hand,face],2)
        return x


class Decoder(nn.Module):
    def __init__(self,num_class,in_channel,length,num_joint,dropout):
        super(Decoder,self).__init__()
        self.fc1 = nn.Linear(num_class,256)
        scale = 16
        self.fc2 = nn.Sequential(
            nn.Linear(256,256*(length//scale)*(32//scale)),
            nn.ReLU(),
            nn.Dropout2d(p=dropout)
        )
        self.deconv6 = nn.Sequential(
            nn.ConvTranspose2d(256,128,3,2,padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=dropout),
        )
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(128,64,3,2,padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=dropout),
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(64,32,3,2,padding=1),
            nn.Dropout2d(p=dropout),
        )
        self.deconv3 = nn.ConvTranspose2d(32,num_joint,3,2,padding=1)
        self.deconv2 = nn.ConvTranspose2d(32,64,(3,1),1,padding=(1,0))
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(64,in_channel,1,1,padding=0),
            nn.ReLU()
            )

    def forward(self,input):
        out = self.fc1(input)
        out = self.fc2(out)

        # After reshape, shape of out is N x J x S x D
        out = out.view(-1,256,2,2)
        out = self.deconv6(out)
        out = self.deconv5(out)

        out = self.deconv4(out)
        out = self.deconv3(out)

        # After permute, shape of out is N x D x T x J
        out = out.permute(0,3,2,1).contiguous()
        out = self.deconv2(out)
        out = self.deconv1(out)

        # After permute ,shape of out is N x T x J x D
        out = out.permute(0,2,3,1)
        return out



