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
from dataset import CSL_Isolated
from train import train
from test import test
import time

# Path setting
data_path = "/home/liweijie/Data/SLR_dataset/S500_color_video"
label_path = "/home/liweijie/Data/SLR_dataset/dictionary.txt"
model_path = "checkpoint"
log_path = "log/cnn3d_{:%Y-%m-%d_%H-%M-%S}.log".format(datetime.now())
sum_path = "runs/slr_cnn3d_{:%Y-%m-%d_%H-%M-%S}".format(datetime.now())

# Log to file & tensorboard writer
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
logger = logging.getLogger('SLR')
logger.info('Logging to file...')
writer = SummaryWriter(os.path.join('runs/', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))

# Use specific gpus
os.environ["CUDA_VISIBLE_DEVICES"]="2"
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparams
num_classes = 500
epochs = 100
batch_size = 8
learning_rate = 1e-5
log_interval = 20
sample_size = 128
sample_duration = 16
drop_p = 0.0
hidden1, hidden2 = 512, 256

best_prec1 = 0.0
store_name = '3Dres_isolated'

# Train with 3DCNN
if __name__ == '__main__':
    # Load data
    transform = transforms.Compose([transforms.Resize([sample_size, sample_size]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])
    trainset = CSL_Isolated(data_path=data_path, label_path=label_path, frames=sample_duration,
        num_classes=num_classes, train=True, transform=transform)
    testset = CSL_Isolated(data_path=data_path, label_path=label_path, frames=sample_duration,
        num_classes=num_classes, train=False, transform=transform)
    logger.info("Dataset samples: {}".format(len(trainset)+len(testset)))
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    # Create model
    model = resnet50(pretrained=True, sample_size=sample_size, sample_duration=sample_duration,
                num_classes=num_classes).to(device)
    # Run the model parallelly
    if torch.cuda.device_count() > 1:
        logger.info("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    # Create loss criterion & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Start training
    logger.info("Training Started".center(60, '#'))
    for epoch in range(epochs):
        # Train the model
        train(model, criterion, optimizer, trainloader, device, epoch, logger, log_interval, writer)
        # Test the model
        prec1 = test(model, criterion, testloader, device, epoch, logger, writer)
        # Save model
        # remember best prec1 and save checkpoint
        is_best = prec1>best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best': best_prec1
        }, is_best, model_path, store_name)
        logger.info("Epoch {} Model Saved".format(epoch+1).center(60, '#'))

    logger.info("Training Finished".center(60, '#'))
