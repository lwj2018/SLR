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
from models.Text2Sign import Encoder,Decoder,Text2Sign
from utils.trainUtils import train_text2sign
from utils.testUtils import test_text2sign
from datasets.CSL_Continuous_Text2Sign import CSL_Continuous_Text2Sign
from args import Arguments
from utils.ioUtils import *
from utils.textUtils import *
from utils.critUtils import LabelSmoothing
from utils.collateUtils import skeleton_collate
from torch.utils.tensorboard import SummaryWriter
import warnings
import pickle
 
warnings.filterwarnings('ignore')

# Path setting
skeleton_root = "/home/haodong/Data/CSL_Continuous_Skeleton"
train_list = "/home/liweijie/Data/public_dataset/train_list.txt"
val_list = "/home/liweijie/Data/public_dataset/val_list.txt"
word2gloss_db = '/home/liweijie/projects/SLR/obj/word2gloss.pkl'
# Hyper params
learning_rate = 1e-5
batch_size = 1
epochs = 1000
hidden_dim = 512
num_classes = 500
clip_length = 32
smoothing = 0.1
stride = 4
# Options
store_name = 'Test_CSL_Text2Sign'
checkpoint = '/home/liweijie/projects/SLR/checkpoint/CSL_Text2Sign_best.pth.tar'
log_interval = 20
device_list = '2'
num_workers = 8

# get arguments
args = Arguments()

# Use specific gpus
os.environ["CUDA_VISIBLE_DEVICES"]=device_list
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use writer to record
writer = SummaryWriter(os.path.join('runs/'+store_name, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))

min_loss = 99999999999.00
start_epoch = 0

# Train with CTC loss
if __name__ == '__main__':
    # Build dictionary
    dictionary = build_dictionary_for_t2s()
    reverse_dict = reverse_dictionary(dictionary)
    vocab_size = len(reverse_dict)
    print("The size of vocabulary is %d"%vocab_size)
    # Load data
    trainset = CSL_Continuous_Text2Sign('train',skeleton_root=skeleton_root,list_file=train_list,dictionary=dictionary)
    valset = CSL_Continuous_Text2Sign('test',skeleton_root=skeleton_root,list_file=train_list,dictionary=dictionary)
    print("Dataset samples: {}".format(len(trainset)+len(valset)))
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    testloader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    # Create model
    encoder = Encoder()
    decoder = Decoder()
    db_file = open(word2gloss_db,'rb')
    database = pickle.load(db_file)
    model = Text2Sign(encoder=encoder, decoder=decoder, word2gloss_database=database).to(device)
    if checkpoint is not None:
        start_epoch, min_loss = resume_model(model,checkpoint)
    # Run the model parallelly
    if torch.cuda.device_count() > 1:
        print("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    # Create loss criterion & optimizer
    criterion = nn.MSELoss()
    # criterion = LabelSmoothing(vocab_size,0,smoothing=smoothing)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Start Test
    print("Test Started".center(60, '#'))
    for epoch in range(start_epoch, start_epoch+1):
        # Test the model
        loss = test_text2sign(model, criterion, testloader, device, epoch, log_interval, writer)

    print("Test Finished".center(60, '#'))



