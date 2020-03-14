import os
import sys
from datetime import datetime
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from models.HCN import hcn
from utils.ioUtils import *
from utils.textUtils import *
from datasets.CSL_Continuous_Openpose import CSL_Continuous_Openpose
import time

# Path setting
skeleton_root = "/home/haodong/Data/CSL_Continious_Skeleton"
train_list = "/home/liweijie/Data/public_dataset/train_list_tmp.txt"
val_list = "/home/liweijie/Data/public_dataset/val_list.txt"
model_path = "checkpoint"

# Use specific gpus
os.environ["CUDA_VISIBLE_DEVICES"]='1'
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparams
num_classes = 500
epochs = 100
batch_size = 1
learning_rate = 1e-5
clip_length = 32
stride = 4
# Options
store_name = '3Dres_isolated'
checkpoint = '/home/liweijie/projects/SLR/checkpoint/HCN_isolated_best.pth.tar'
log_interval = 20

best_prec1 = 0.0

writer = SummaryWriter(os.path.join('runs/', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))

# Train with 3DCNN
if __name__ == '__main__':
    # Build dictionary
    dictionary = build_isl_dictionary()
    reverse_dict = reverse_dictionary(dictionary)
    # Load data
    trainset = CSL_Continuous_Openpose(skeleton_root=skeleton_root,list_file=train_list,dictionary=dictionary,
            clip_length=clip_length,stride=stride)
    # testset = CSL_Continuous_Openpose(skeleton_root=skeleton_root,list_file=val_list,dictionary=dictionary,
    #         clip_length=clip_length,stride=stride)
    print("Trainset samples: {}".format(len(trainset)))
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    # testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    # Create model
    model = hcn(num_class=num_classes).to(device)
    if checkpoint is not None:
        start_epoch, best_wer = resume_model(model,checkpoint)
    # Run the model parallelly
    if torch.cuda.device_count() > 1:
        print("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    for i,batch in enumerate(trainloader):
        if i==0:
            input, tgt = batch['input'], batch['tgt']
            # Shape of input is: N x S x clip_length x J x D
            N,S,l,J,D = input.size()
            input = input.to(device)
            # After view & permute, shape of input is: N x D x (Sxl) x J
            input = input.view(N,-1,J,D).permute(0,3,1,2)
            input = F.upsample(input,size=(2*S*l,J),mode='bilinear').contiguous()
            # After view, shape of input is: N x D x (2xS) x l x J
            input = input.view(N,D,-1,l,J)
            # After permute, shape of input is: N x (2xS) x l x J x D
            input = input.permute(0,2,3,4,1)
            input = input.view( (-1,) + input.size()[-3:] )
            output = model(input)
            output = output.argmax(1)
            output = output.data.cpu().numpy()
            tgt = tgt.view(-1)
            tgt = tgt.data.cpu().numpy()
            print("output: {}".format(output))
            print("tgt: {}".format(tgt))
            output = ' '.join(itos(output,reverse_dict))
            tgt = ' '.join(itos(tgt,reverse_dict))
            writer.add_text('outputs', 
                            str(output),
                            0)
            writer.add_text('tgt',
                            str(tgt),
                            0)