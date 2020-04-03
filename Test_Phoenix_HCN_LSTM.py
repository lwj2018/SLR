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
from datasets.CSL_Phoenix_Openpose import CSL_Phoenix_Openpose
from args import Arguments
from utils.ioUtils import *
from utils.textUtils import *
from utils.critUtils import LabelSmoothing
from utils.collateUtils import skeleton_collate
from torch.utils.tensorboard import SummaryWriter
import warnings
 
warnings.filterwarnings('ignore')

# Path setting
train_skeleton_root = "/home/haodong/Data/multisigner/train"
train_annotation_file = "/mnt/data/public/datasets/phoenix2014-release/phoenix-2014-multisigner/annotations/manual/train.corpus.csv"
dev_skeleton_root = "/home/haodong/Data/multisigner/dev"
dev_annotation_file = "/mnt/data/public/datasets/phoenix2014-release/phoenix-2014-multisigner/annotations/manual/dev.corpus.csv"
test_skeleton_root = "/home/haodong/Data/multisigner/test"
test_annotation_file = "/mnt/data/public/datasets/phoenix2014-release/phoenix-2014-multisigner/annotations/manual/test.corpus.csv"
# Hyper params
learning_rate = 1e-6
batch_size = 2
epochs = 1000
hidden_dim = 512
num_classes = 500
clip_length = 32
smoothing = 0.1
stride = 4
clip_g = 1
# Options
checkpoint = '/home/liweijie/projects/SLR/checkpoint/20200401_Phoenix_HCN_LSTM_checkpoint.pth.tar'
log_interval = 5
device_list = '1'
num_workers = 8

# get arguments
args = Arguments()

# Use specific gpus
os.environ["CUDA_VISIBLE_DEVICES"]=device_list
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use writer to record
writer = SummaryWriter(os.path.join('runs/test_phoenix_hcn_lstm', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))

best_wer = 999.00
start_epoch = 0

# Train with CTC loss
if __name__ == '__main__':
    # Build dictionary
    dictionary = build_isl_dictionary[train_annotation_file)
    reverse_dict = reverse_phoenix_dictionary(dictionary)
    vocab_size = len(dictionary)
    print("The size of vocabulary is %d"%vocab_size)
    # Load data
    trainset = CSL_Phoenix_Openpose(skeleton_root=train_skeleton_root,annotation_file=train_annotation_file,dictionary=dictionary,
            clip_length=clip_length,stride=stride)
    devset = CSL_Phoenix_Openpose(skeleton_root=dev_skeleton_root,annotation_file=dev_annotation_file,dictionary=dictionary,
            clip_length=clip_length,stride=stride)
    testset = CSL_Phoenix_Openpose(skeleton_root=test_skeleton_root,annotation_file=test_annotation_file,dictionary=dictionary,
            clip_length=clip_length,stride=stride)
    print("Dataset samples: {}".format(len(trainset)+len(devset)))
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,
            collate_fn=skeleton_collate)
    devloader = DataLoader(devset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,
            collate_fn=skeleton_collate)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,
            collate_fn=skeleton_collate)
    # Create model
    model = hcn_lstm(vocab_size,clip_length=clip_length,
                num_classes=num_classes,hidden_dim=hidden_dim).to(device)
    start_epoch, best_wer = resume_model(model, checkpoint)
    # Run the model parallelly
    if torch.cuda.device_count() > 1:
        print("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    # Create loss criterion
    criterion = nn.CTCLoss() 

    wer = 100.00
    # Start training
    print("Evaluation Started".center(60, '#'))
    for epoch in range(start_epoch, start_epoch+1):
        # Test the model
        wer = test_hcn_lstm(model, criterion, trainloader, device, epoch, log_interval, writer, reverse_dict)

    print("Evaluation Finished".center(60, '#'))



