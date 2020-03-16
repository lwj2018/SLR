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
from models.Transformer import CSL_Transformer
from utils.trainUtils import train
from utils.testUtils import test
from datasets.CSL_Phoenix_RGB import CSL_Phoenix_RGB
from args import Arguments
from utils.ioUtils import save_checkpoint, resume_model
from utils.critUtils import LabelSmoothing
from utils.textUtils import *
from utils.collateUtils import rgb_collate
from torch.utils.tensorboard import SummaryWriter
import warnings
 
warnings.filterwarnings('ignore')
# Path setting
train_frame_root = "/mnt/data/public/datasets/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px/train"
train_annotation_file = "/mnt/data/public/datasets/phoenix2014-release/phoenix-2014-multisigner/annotations/manual/train.corpus.csv"
dev_frame_root = "/mnt/data/public/datasets/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px/dev"
dev_annotation_file = "/mnt/data/public/datasets/phoenix2014-release/phoenix-2014-multisigner/annotations/manual/dev.corpus.csv"
# Hyper params
learning_rate = 1e-6
batch_size = 2
epochs = 1000
sample_size = 128
num_classes = 512
clip_length = 16
smoothing = 0.1
stride = 8
# Options
store_name = 'Transformer_3dres'
checkpoint = None
log_interval = 100

# get arguments
args = Arguments()

# Use specific gpus
os.environ["CUDA_VISIBLE_DEVICES"]=args.device_list
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use writer to record
writer = SummaryWriter(os.path.join('runs/', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))

best_wer = 999.00
start_epoch = 0


# Train with Transformer
if __name__ == '__main__':
    # build dictionary
    dictionary = build_dictionary([train_annotation_file,dev_annotation_file])
    reverse_dict = reverse_dictionary(dictionary)
    vocab_size = len(dictionary)
    print("The size of vocabulary is %d"%vocab_size)
    # Load data
    transform = transforms.Compose([transforms.Resize([sample_size, sample_size]),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5], std=[0.5])])
    trainset = CSL_Phoenix_RGB(frame_root=train_frame_root,annotation_file=train_annotation_file,
            transform=transform,dictionary=dictionary,clip_length=clip_length,stride=stride)
    devset = CSL_Phoenix_RGB(frame_root=dev_frame_root,annotation_file=dev_annotation_file,
            transform=transform,dictionary=dictionary,clip_length=clip_length,stride=stride)
    print("Dataset samples: {}".format(len(trainset)+len(devset)))
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True,
            collate_fn=rgb_collate)
    testloader = DataLoader(devset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True,
            collate_fn=rgb_collate)
    # Create model
    model = CSL_Transformer(vocab_size,vocab_size,sample_size=sample_size, clip_length=clip_length,
                num_classes=num_classes,modal='rgb').to(device)
    if checkpoint is not None:
        start_epoch, best_wer = resume_model(model,checkpoint)
    # Run the model parallelly
    if torch.cuda.device_count() > 1:
        print("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    # Create loss criterion & optimizer
    # criterion = nn.CrossEntropyLoss()
    criterion = LabelSmoothing(vocab_size,0,smoothing=smoothing)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Start training
    print("Training Started".center(60, '#'))
    for epoch in range(start_epoch, epochs):
        # Train the model
        train(model, criterion, optimizer, trainloader, device, epoch, log_interval, writer, reverse_dict)
        # Test the model
        wer = test(model, criterion, testloader, device, epoch, log_interval, writer, reverse_dict)
        # Save model
        # remember best wer and save checkpoint
        is_best = wer<best_wer
        best_wer = min(wer, best_wer)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_wer': best_wer
        }, is_best, args.model_path, store_name)
        print("Epoch {} Model Saved".format(epoch+1).center(60, '#'))

    print("Training Finished".center(60, '#'))

