import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.nn import functional as F

def select_indices(x):
    left_arm = x[:,:,[3,4],:]
    right_arm = x[:,:,[6,7],:]
    face = x[:,:,25:95,:]
    left_hand = x[:,:,95:116,:]
    right_hand = x[:,:,116:137,:]
    x = torch.cat([left_arm,right_arm,left_hand,right_hand,face],dim=2)
    return x


class lstm(nn.Module):
    def __init__(self,input_size,
                hidden_size=512,
                hidden_dim=512,
                num_layers=3,
                dropout_rate=0.1,
                num_classes=500,
                bidirectional=True):
        super(lstm,self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional)
        if bidirectional: hidden_size = 2*hidden_size
        self.fc1 = nn.Linear(hidden_size,hidden_dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(hidden_dim,num_classes)

    def forward(self,x):
        # Flatten
        x = select_indices(x)
        x = x.flatten(start_dim=2)
        # RNN forward
        x, (h_n,c_n) = self.lstm(x)
        # Avg pool
        x = x.mean(dim=1)
        # FC forward
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class gru(nn.Module):
    def __init__(self,input_size,
                hidden_size=512,
                hidden_dim=512,
                num_layers=3,
                dropout_rate=0.1,
                num_classes=500,
                bidirectional=True):
        super(gru,self).__init__()
        self.gru = nn.GRU(input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional)
        if bidirectional: hidden_size = 2*hidden_size
        self.fc1 = nn.Linear(hidden_size,hidden_dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(hidden_dim,num_classes)

    def forward(self,x):
        # Flatten
        x = select_indices(x)
        x = x.flatten(start_dim=2)
        # RNN forward
        x, (h_n,c_n) = self.gru(x)
        # Avg pool
        x = x.mean(dim=1)
        # FC forward
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
