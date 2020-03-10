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
from models.Conv3D import CNN3D, resnet18, resnet34, r3d_18, resnet50
from utils.ioUtils import *
from utils.textUtils import *
from datasets.CSL_Continuous_RGB import CSL_Continuous_RGB
import time

# Path setting
frame_root = "/home/liweijie/Data/public_dataset/color_frame"
train_list = "/home/liweijie/Data/public_dataset/train_list.txt"
val_list = "/home/liweijie/Data/public_dataset/val_list.txt"
writer_path = "runs/test_isl_3dres_on_csl"
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
log_interval = 20
sample_size = 128
sample_duration = 16
drop_p = 0.0
hidden1, hidden2 = 512, 256
# Options
store_name = '3Dres_isolated'
checkpoint = '/home/liweijie/projects/SLR/checkpoint/3Dres_isolated_checkpoint.pth.tar'

best_prec1 = 0.0

create_path(writer_path)
writer = SummaryWriter(os.path.join(writer_path, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))

# Train with 3DCNN
if __name__ == '__main__':
    # Build dictionary
    dictionary = build_isl_dictionary()
    reverse_dict = reverse_dictionary(dictionary)
    # Load data
    transform = transforms.Compose([transforms.Resize([sample_size, sample_size]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])
    trainset = CSL_Continuous_RGB(frame_root=frame_root,list_file=train_list,transform=transform,
            dictionary=dictionary)
    testset = CSL_Continuous_RGB(frame_root=frame_root,list_file=val_list,transform=transform,
            dictionary=dictionary)
    print("Dataset samples: {}".format(len(trainset)+len(testset)))
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    # Create model
    model = resnet50(pretrained=True, sample_size=sample_size, sample_duration=sample_duration,
                num_classes=num_classes).to(device)
    if checkpoint is not None:
        start_epoch, best_wer = resume_model(model,checkpoint)
    # Run the model parallelly
    if torch.cuda.device_count() > 1:
        print("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    for i,batch in enumerate(trainloader):
        if i==0:
            input, tgt = batch['input'], batch['tgt']
            input = input.to(device)
            input = input.view( (-1,) + input.size()[-4:] )
            output = model(input)
            output = output.argmax(1)
            output = output.data.cpu().numpy()
            tgt = tgt.view(-1)
            tgt = tgt.data.cpu().numpy()
            output = ''.join(itos(output,reverse_dict))
            tgt = ''.join(itos(tgt,reverse_dict))
            writer.add_text('outputs', 
                            str(output),
                            0)
            writer.add_text('tgt',
                            str(tgt),
                            0)