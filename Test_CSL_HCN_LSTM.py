import os
import sys
import time
from datetime import datetime
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from models.HCN_LSTM import hcn_lstm
from utils.trainUtils import train_hcn_lstm
from utils.testUtils import test_hcn_lstm
from datasets.CSL_Continuous_Openpose import CSL_Continuous_Openpose
from args import Arguments
from utils.ioUtils import *
from utils.textUtils import *
from utils.critUtils import LabelSmoothing
from utils.collateUtils import skeleton_collate
from torch.utils.tensorboard import SummaryWriter
import warnings
 
warnings.filterwarnings('ignore')

# Path setting
skeleton_root = "/mnt/data/haodong/CSL_Continious_Skeleton"
train_list = "/home/liweijie/Data/public_dataset/train_list.txt"
val_list = "/home/liweijie/Data/public_dataset/val_list.txt"
# Hyper params
learning_rate = 1e-5
batch_size = 2
epochs = 1000
hidden_dim = 512
num_classes = 500
clip_length = 32
smoothing = 0.1
stride = 4
# Options
store_name = 'HCN_LSTM'
checkpoint = '/home/liweijie/projects/SLR/checkpoint/HCN_LSTM_checkpoint.pth.tar'
log_interval = 100
device_list = '1'
num_workers = 0

# get arguments
args = Arguments()

# Use specific gpus
os.environ["CUDA_VISIBLE_DEVICES"]=device_list
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use writer to record
writer = SummaryWriter(os.path.join('runs/', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))

best_wer = 999.00
start_epoch = 0

# Train with Transformer
if __name__ == '__main__':
    # Build dictionary
    dictionary = build_csl_dictionary()
    reverse_dict = reverse_dictionary(dictionary)
    vocab_size = len(dictionary)
    print("The size of vocabulary is %d"%vocab_size)
    # Load data
    trainset = CSL_Continuous_Openpose(skeleton_root=skeleton_root,list_file=train_list,dictionary=dictionary,
            clip_length=clip_length,stride=stride,add_two_end=False)
    valset = CSL_Continuous_Openpose(skeleton_root=skeleton_root,list_file=val_list,dictionary=dictionary,
            clip_length=clip_length,stride=stride,add_two_end=False)
    print("Dataset samples: {}".format(len(trainset)+len(valset)))
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,
            collate_fn=skeleton_collate)
    testloader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,
            collate_fn=skeleton_collate)
    # Create model
    model = hcn_lstm(vocab_size,clip_length=clip_length,
                num_classes=num_classes,hidden_dim=hidden_dim).to(device)
    if checkpoint is not None:
        start_epoch, best_wer = resume_model(model,checkpoint)
    # Run the model parallelly
    if torch.cuda.device_count() > 1:
        print("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    # Create loss criterion & optimizer
    criterion = nn.CTCLoss()

    # Start Evaluation
    print("Evaluation Started".center(60, '#'))
    for epoch in range(start_epoch, start_epoch+1):
        # Test the model
        wer = test_hcn_lstm(model, criterion, trainloader, device, epoch, log_interval, writer, reverse_dict)

    print("Evaluation Finished".center(60, '#'))



