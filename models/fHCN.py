import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.nn import functional as F
from models import Conv3D, HCN

class fHCN(nn.Module):

    def __init__(self, num_class, 
                        length=32,
                        length=16,
                        sample_size=224,
                        rgb_feature_dim=256):
        # T N D
        super(fHCN, self).__init__()
        self.num_class = num_class
        self.length = length
        self.length = length
        self.sample_size = sample_size
        self.rgb_feature_dim = rgb_feature_dim

        self.get_skeleton_model()
        self.get_cnn_model()
        self.simple_fusion = simple_fusion(3*rgb_feature_dim+num_class,num_class)


    def get_cnn_model(self):
        self.cnn_model = Conv3D.resnet18(pretrained=True, sample_size=self.sample_size, 
                        sample_duration=self.length, num_classes=self.rgb_feature_dim)

    def get_skeleton_model(self):
        self.skeleton_model = hcn(self.num_class)
    

    def forward(self, input, image):
        '''
            shape of input is: N x J x T x D
            shape of image is: N x C[lefthand(R G B)...] x H x W
                            lefthand righthand   face    ...
                             (R G B)  (R G B)   (R G B)  ...
        '''
        out = self.skeleton_model(input)
        out = torch.norm(out,1,dim=1)

        size = image.size()
        # N x 16 x 3 x C x H x W
        image = image.view(size[0],self.length,-1,3,size[-2],size[-1])
        # N x 3 x C x 16 x H x W
        image = image.permute(0,2,3,1,4,5)
        # now shape of image is N(3x) x C x 16 x H x W
        image = image.view( (-1,) + image.size()[-4:] )
        out_c = self.cnn_model(image)
        out_c = out_c.view(size[0],-1)
        out_c = torch.norm(out_c,1,dim=1)

        # shape of out is N x C1, C1=?
        # shape of out_c is N x C2
        out = torch.stack([out,out_c],2)
        out = self.simple_fusion(out)

        return out

class simple_fusion(nn.Module):
    def __init__(self,in_channel,num_class):
        super(simple_fusion,self).__init__()
        self.fusion1 = nn.Sequential(
            nn.Linear(in_channel,500),
            nn.ReLU(),
            nn.Dropout(p=0.5))
        self.fusion2 = nn.Linear(500,num_class)

    def forward(self,input):
        N = input.size(0)
        out = input.view(N,-1)
        out = self.fusion1(out)
        out = self.fusion2(out)
        return out


