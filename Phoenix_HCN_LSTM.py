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
from utils.testUtils import eval_hcn_lstm,test_hcn_lstm
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
learning_rate = 1e-5
batch_size = 4
epochs = 1000
hidden_dim = 512
num_classes = 500
clip_length = 32
smoothing = 0.1
stride = 4
clip_g = 1
# Options
store_name = 'Phoenix_HCN_LSTM'
hcn_checkpoint = "/home/liweijie/projects/SLR/checkpoint/20200315_82.106_HCN_isolated_best.pth.tar"
# hcn_lstm_ckpt = '/home/liweijie/projects/SLR/checkpoint/20200318_HCN_LSTM_best.pth.tar'
hcn_lstm_ckpt = '/home/liweijie/projects/SLR/checkpoint/20200401_Phoenix_HCN_LSTM_best.pth.tar'
checkpoint = None#'/home/liweijie/projects/SLR/checkpoint/Phoenix_HCN_LSTM_checkpoint.pth.tar'
log_interval = 100
device_list = '3'
num_workers = 8

# get arguments
args = Arguments()

# Use specific gpus
os.environ["CUDA_VISIBLE_DEVICES"]=device_list
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use writer to record
writer = SummaryWriter(os.path.join('runs/phoenix_hcn_lstm', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))

best_wer = 999.00
start_epoch = 0

# Train with CTC loss
if __name__ == '__main__':
    # Build dictionary
    dictionary = build_dictionary(train_annotation_file)
    reverse_dict = reverse_phoenix_dictionary(dictionary)
    vocab_size = len(reverse_dict)
    print("The size of vocabulary is %d"%vocab_size)
    # Load data
    trainset = CSL_Phoenix_Openpose(skeleton_root=train_skeleton_root,annotation_file=train_annotation_file,dictionary=dictionary,
            clip_length=clip_length,stride=stride)
    devset = CSL_Phoenix_Openpose(skeleton_root=dev_skeleton_root,annotation_file=dev_annotation_file,dictionary=dictionary,
            clip_length=clip_length,stride=stride)
    testset = CSL_Phoenix_Openpose(skeleton_root=test_skeleton_root,annotation_file=test_annotation_file,dictionary=dictionary,
            clip_length=clip_length,stride=stride)
    print("Dataset samples: {}".format(len(trainset)+len(devset)+len(testset)))
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,
            collate_fn=skeleton_collate)
    devloader = DataLoader(devset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,
            collate_fn=skeleton_collate)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,
            collate_fn=skeleton_collate)
    # Create model
    model = hcn_lstm(vocab_size,clip_length=clip_length,
                num_classes=num_classes,hidden_dim=hidden_dim,dropout=0.4).to(device)
    model = resume_hcn_module(model, hcn_checkpoint)
    if hcn_lstm_ckpt is not None:
        model = resume_hcn_lstm(model,hcn_lstm_ckpt)
    if checkpoint is not None:
        start_epoch, best_wer = resume_model(model, checkpoint)
    # Run the model parallelly
    if torch.cuda.device_count() > 1:
        print("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    # Create loss criterion & optimizer
    # criterion = nn.CTCLoss(zero_infinity=True)
    criterion = nn.CTCLoss() 
    # criterion = LabelSmoothing(vocab_size,0,smoothing=smoothing)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    wer = 100.00
    # Start training
    print("Training Started".center(60, '#'))
    for epoch in range(start_epoch, epochs):
        # Train the model
        train_hcn_lstm(model, criterion, optimizer, trainloader, device, epoch, log_interval, writer, reverse_dict, clip_g)
        # Eval the model
        eval_hcn_lstm(model, criterion, devloader, device, epoch, log_interval, writer, reverse_dict)
        # Test the model
        wer = test_hcn_lstm(model, criterion, testloader, device, epoch, log_interval, writer, reverse_dict)
        # Save model
        # remember best wer and save checkpoint
        is_best = wer<best_wer
        best_wer = min(wer, best_wer)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best': best_wer
        }, is_best, args.model_path, store_name)
        print("Epoch {} Model Saved".format(epoch+1).center(60, '#'))

    print("Training Finished".center(60, '#'))



