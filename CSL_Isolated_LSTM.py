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
from models.LSTM import lstm,gru
from utils.trainUtils import train_isolated
from utils.testUtils import test_isolated
from datasets.CSL_Isolated_Openpose import CSL_Isolated_Openpose
from args import Arguments
from utils.ioUtils import *
from utils.critUtils import LabelSmoothing
from utils.textUtils import build_dictionary, reverse_dictionary
from torch.utils.tensorboard import SummaryWriter

# Path settings
skeleton_root = "/home/liweijie/skeletons_dataset"
train_file = "input/train_list.txt"
val_file = "input/val_list.txt"
# Hyper params
learning_rate = 1e-5
batch_size = 16
epochs = 1000
num_class = 500
num_joints = 116
length = 32
dropout = 0.1
# Options
store_name = 'LSTM_isolated'
summary_name = 'runs/' + store_name
checkpoint = None
device_list = '1'
log_interval = 100

# Get arguments
args = Arguments()

# Use specific gpus
os.environ["CUDA_VISIBLE_DEVICES"]=device_list
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use writer to record
writer = SummaryWriter(os.path.join(summary_name, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))

best_prec1 = 0.0
start_epoch = 0

# Train with Transformer
if __name__ == '__main__':
    # Load data
    trainset = CSL_Isolated_Openpose(skeleton_root=skeleton_root,list_file=train_file,
        length=length)
    devset = CSL_Isolated_Openpose(skeleton_root=skeleton_root,list_file=val_file,
        length=length)
    print("Dataset samples: {}".format(len(trainset)+len(devset)))
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    testloader = DataLoader(devset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    # Create model
    model = lstm(input_size=num_joints*2,hidden_size=512,hidden_dim=512,
        num_layers=3,dropout_rate=dropout,num_classes=num_class,
        bidirectional=True).to(device)
    if checkpoint is not None:
        start_epoch, best_prec1 = resume_model(model,checkpoint)
    # Run the model parallelly
    if torch.cuda.device_count() > 1:
        print("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    # Create loss criterion & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Start training
    print("Training Started".center(60, '#'))
    for epoch in range(start_epoch, epochs):
        # Train the model
        train_isolated(model, criterion, optimizer, trainloader, device, epoch, log_interval, writer)
        # Test the model
        prec1 = test_isolated(model, criterion, testloader, device, epoch, log_interval, writer)
        # Save model
        # remember best prec1 and save checkpoint
        is_best = prec1>best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best': best_prec1
        }, is_best, args.model_path, store_name)
        print("Epoch {} Model Saved".format(epoch+1).center(60, '#'))

    print("Training Finished".center(60, '#'))


