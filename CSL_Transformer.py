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
from datasets.CSL_Phoenix import CSL_Phoenix
from datasets.CSL_Phoenix_Skeleton import CSL_Phoenix_Skeleton
from args import Arguments
from utils.ioUtils import save_checkpoint, resume_model
from utils.critUtils import LabelSmoothing
from utils.textUtils import build_dictionary, reverse_dictionary
from torch.utils.tensorboard import SummaryWriter

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
    dictionary = build_dictionary([args.train_annotation_file,args.dev_annotation_file])
    reverse_dict = reverse_dictionary(dictionary)
    vocab_size = len(dictionary)
    print("The size of vocabulary is %d"%vocab_size)
    # Load data
    if args.modal=='rgb':
        transform = transforms.Compose([transforms.Resize([args.sample_size, args.sample_size]),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5], std=[0.5])])
        trainset = CSL_Phoenix(frame_root=args.train_frame_root,annotation_file=args.train_annotation_file,
            transform=transform,dictionary=dictionary)
        devset = CSL_Phoenix(frame_root=args.dev_frame_root,annotation_file=args.dev_annotation_file,
            transform=transform,dictionary=dictionary)
    elif args.modal=='skeleton':
        trainset = CSL_Phoenix_Skeleton(skeleton_root=args.train_skeleton_root,annotation_file=args.train_annotation_file,
            dictionary=dictionary,clip_length=args.clip_length,stride=args.stride)
        devset = CSL_Phoenix_Skeleton(skeleton_root=args.dev_skeleton_root,annotation_file=args.dev_annotation_file,
            dictionary=dictionary,clip_length=args.clip_length,stride=args.stride)
    print("Dataset samples: {}".format(len(trainset)+len(devset)))
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
    testloader = DataLoader(devset, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)
    # Create model
    model = CSL_Transformer(vocab_size,vocab_size,sample_size=args.sample_size, clip_length=args.clip_length,
                num_classes=args.num_classes,modal=args.modal).to(device)
    if args.resume_model is not None:
        start_epoch, best_wer = resume_model(model,args.resume_model)
    # Run the model parallelly
    if torch.cuda.device_count() > 1:
        print("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    # Create loss criterion & optimizer
    # criterion = nn.CrossEntropyLoss()
    criterion = LabelSmoothing(vocab_size,0,smoothing=args.smoothing)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Start training
    print("Training Started".center(60, '#'))
    for epoch in range(start_epoch, args.epochs):
        # Train the model
        train(model, criterion, optimizer, trainloader, device, epoch, args.log_interval, writer, reverse_dict)
        # Test the model
        wer = test(model, criterion, testloader, device, epoch, args.log_interval, writer, reverse_dict)
        # Save model
        # remember best wer and save checkpoint
        is_best = wer<best_wer
        best_wer = min(wer, best_wer)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_wer': best_wer
        }, is_best, args.model_path, args.store_name)
        print("Epoch {} Model Saved".format(epoch+1).center(60, '#'))

    print("Training Finished".center(60, '#'))

