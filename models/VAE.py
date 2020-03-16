import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self,num_class, in_channel=2,
                            length=32,
                            num_joint=25+2*21+70,
                            dropout=0.2,
                            latent_dim=256,
                            conv_feature_dim=1024):
        super(VAE, self).__init__()
        self.encoder = Encoder(num_class,in_channel,length,num_joint,dropout)
        self.decoder = Decoder(num_class,in_channel,length,num_joint,dropout)
        self.classifier = nn.Linear(conv_feature_dim,num_class)
        self.fc_mu = nn.Linear(conv_feature_dim,latent_dim)
        self.fc_var = nn.Linear(conv_feature_dim,latent_dim)
        self.decoder_input = nn.Linear(latent_dim,conv_feature_dim)

    def encode(self,input):
        out = self.encoder(input)
        out = torch.flatten(out,start_dim=1)
        mu = self.fc_mu(out)
        log_var = self.fc_var(out)
        return [mu,log_var]

    def decode(self,input):
        out = self.decoder_input(input)
        # After reshape, shape of out is N x J x S x D
        out = out.view(-1,256,2,2)
        out = self.decoder(out)
        return out

    def classify(self,input):
        out = self.encoder(input)
        out = torch.flatten(out,start_dim=1)
        out = self.classifier(out)
        return out

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [N x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [N x D]
        :return: (Tensor) [N x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self,x):
        mu,log_var = self.encode(x)
        z = self.reparameterize(mu,log_var)
        return [self.decode(z),x,mu,log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x, **kwargs):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [N x C x H x W]
        :return: (Tensor) [N x C x H x W]
        """

        return self.forward(x)[0]


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

    def forward(self,input):
        # shape of input is: N x T x J x D
        # After permute, shape of input is N x D x T x J
        input = input.permute(0,3,1,2)
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

        return out



class Decoder(nn.Module):
    def __init__(self,num_class,in_channel,length,num_joint,dropout):
        super(Decoder,self).__init__()
        self.deconv6 = nn.Sequential(
            nn.ConvTranspose2d(256,128,3,2,padding=1,output_padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=dropout),
        )
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(128,64,3,2,padding=1,output_padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=dropout),
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(64,32,3,2,padding=1,output_padding=1),
            nn.Dropout2d(p=dropout),
        )
        self.deconv3 = nn.ConvTranspose2d(32,num_joint,3,2,padding=1,output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(32,64,(3,1),1,padding=(1,0))
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(64,in_channel,1,1,padding=0),
            nn.ReLU()
            )

    def forward(self,input):
        # shape of input is N x J x S x D
        out = self.deconv6(input)
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



