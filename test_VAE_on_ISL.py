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
from models.VAE import VAE
from utils.trainUtils import train_vae
from utils.testUtils import test_vae
from datasets.CSL_Isolated_Openpose import CSL_Isolated_Openpose
from args import Arguments
from utils.ioUtils import *
from utils.critUtils import LabelSmoothing
from utils.textUtils import build_dictionary, reverse_dictionary
from torch.utils.tensorboard import SummaryWriter
import warnings
 
warnings.filterwarnings('ignore')
# Path settings
video_root = "/home/liweijie/SLR_dataset/S500_color_video"
skeleton_root = "/home/liweijie/skeletons_dataset"
train_file = "input/train_list.txt"
val_file = "input/val_list.txt"
# Hyper params
learning_rate = 1e-5
batch_size = 2
epochs = 1000
sample_size = 224
num_class = 500
length = 32
dropout = 0.2
# Options
store_name = 'VAE_isolated'
checkpoint = '/home/liweijie/projects/SLR/checkpoint/VAE_isolated_checkpoint.pth.tar'
log_interval = 100
device_list = '1'
output_path = 'obj/vae_generate_ISL'
is_csl = False
num_workers = 8

# Get arguments
args = Arguments()

# Use specific gpus
os.environ["CUDA_VISIBLE_DEVICES"]=device_list
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    testloader = DataLoader(devset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    # Create model
    model = VAE(num_class,dropout=dropout).to(device)
    if checkpoint is not None:
        start_epoch, best_prec1 = resume_model(model,checkpoint)
    # Run the model parallelly
    if torch.cuda.device_count() > 1:
        print("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    # Create loss criterion & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Start Test
    print("Test Started".center(60, '#'))
    for epoch in range(start_epoch, start_epoch+1):
        # Test the model
        test_vae(model, criterion, testloader, device, epoch, log_interval, output_path, is_csl)

    print("Test Finished".center(60, '#'))


