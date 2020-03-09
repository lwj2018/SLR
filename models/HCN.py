import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.nn import functional as F

class hcn(nn.Module):
    def __init__(self,num_class, in_channel=2,
                            length=32,
                            num_joint=10,
                            dropout=0.2):
        super(hcn, self).__init__()
        self.num_class = num_class
        self.in_channel = in_channel
        self.length = length
        self.num_joint = num_joint
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel,64,1,1,padding=0),
            nn.ReLU()
            )
        self.conv2 = nn.Conv2d(64,32,(3,1),1,padding=(1,0))
        self.hconv = HierarchyConv()
        self.conv4 = nn.Sequential(
            nn.Conv2d(32,64,3,1,padding=1),
            nn.Dropout2d(p=dropout),
            nn.MaxPool2d(2)
        )

        self.convm1 = nn.Sequential(
            nn.Conv2d(in_channel,64,1,1,padding=0),
            nn.ReLU()
            )
        self.convm2 = nn.Conv2d(64,32,(3,1),1,padding=(1,0))
        self.hconvm = HierarchyConv()
        self.convm4 = nn.Sequential(
            nn.Conv2d(32,64,3,1,padding=1),
            nn.Dropout2d(p=dropout),
            nn.MaxPool2d(2)
        )
                
        self.conv5 = nn.Sequential(
            nn.Conv2d(128,128,3,1,padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=dropout),
            nn.MaxPool2d(2)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128,256,3,1,padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=dropout),
            nn.MaxPool2d(2)
        )

        # scale related to total number of maxpool layer
        scale = 16
        self.fc7 = nn.Sequential(
            nn.Linear(256*(length//scale)*(32//scale),256),
            nn.ReLU(),
            nn.Dropout2d(p=dropout)
        )
        self.fc8 = nn.Linear(256,self.num_class)

    def forward(self,input):
        output = self.get_feature(input)
        output = self.classify(output)
        return output

    def get_feature(self,input):
        # input: N T J D
        input = input.permute(0,3,1,2)
        N, D, T, J = input.size()
        motion = input[:,:,1::,:]-input[:,:,0:-1,:]
        motion = F.upsample(motion,size=(T,J),mode='bilinear').contiguous()

        out = self.conv1(input)
        out = self.conv2(out)
        out = out.permute(0,3,2,1).contiguous()
        # out: N J T D
        
        # out = self.conv3(out)
        out = self.hconv(out)
        out = self.conv4(out)

        outm = self.convm1(motion)
        outm = self.convm2(outm)
        outm = outm.permute(0,3,2,1).contiguous()
        # outm: N J T D

        # outm = self.convm3(outm)
        outm = self.hconvm(outm)
        outm = self.convm4(outm)

        out = torch.cat((out,outm),dim=1)
        out = self.conv5(out)
        out = self.conv6(out)
        # out:  N J T(T/16) D
        return out

    def classify(self,input):
        out = input.view(input.size(0),-1)
        out = self.fc7(out)
        out = self.fc8(out)

        t = out
        # assert not ((t != t).any())# find out nan in tensor
        assert not (t.abs().sum() == 0) # find out 0 tensor
        # N x C (num_class)
        return out

class HierarchyConv(nn.Module):
    def __init__(self):
        super(HierarchyConv,self).__init__()
        self.convla = nn.Conv2d(2,16,3,1,padding=1)
        self.convra = nn.Conv2d(2,16,3,1,padding=1)
        self.conflh = nn.Conv2d(21,16,3,1,padding=1)
        self.confrh = nn.Conv2d(21,16,3,1,padding=1)
        self.convf = nn.Conv2d(70,32,3,1,padding=1)
        self.convl = nn.Conv2d(32,32,3,1,padding=1)
        self.convr = nn.Conv2d(32,32,3,1,padding=1)
        self.parts = 3
        self.conv = nn.Sequential(
            nn.Conv2d(self.parts*32,32,3,1,padding=1),
            nn.MaxPool2d(2)
        )

    def forward(self,input):
        left_arm = input[:,[3,4],:,:]
        right_arm = input[:,[6,7],:,:]
        face = input[:,25:95,:,:]
        left_hand = input[:,95:116,:,:]
        right_hand = input[:,116:137,:,:]
        # left_arm = input[:,[0,1],:,:]
        # right_arm = input[:,[2,3],:,:]
        # face = input[:,4:74,:,:]
        # left_hand = input[:,74:95,:,:]
        # right_hand = input[:,95:116,:,:]
        l1 = self.convla(left_arm) 
        r1 = self.convra(right_arm) 
        l2 = self.conflh(left_hand)
        r2 = self.confrh(right_hand)
        l = torch.cat([l1,l2],1)
        r = torch.cat([r1,r2],1)
        l = self.convl(l)
        r = self.convr(r)
        f = self.convf(face)
        out = torch.cat([l,r,f],1)
        out = self.conv(out)
        return out