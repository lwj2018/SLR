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

# Log to file
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.FileHandler(args.log_path)])
logger = logging.getLogger('SLR')
logger.info('Logging to file...')

# Use specific gpus
os.environ["CUDA_VISIBLE_DEVICES"]=args.device_list
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dont use writer temporarily
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
    logger.info("Dataset samples: {}".format(len(trainset)+len(devset)))
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)
    testloader = DataLoader(devset, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)
    # Create model
    model = CSL_Transformer(vocab_size,vocab_size,sample_size=args.sample_size, clip_length=args.clip_length,
                num_classes=args.num_classes,modal=args.modal).to(device)
    if args.resume_model is not None:
        start_epoch, best_wer = resume_model(model,args.resume_model)
    # Run the model parallelly
    if torch.cuda.device_count() > 1:
        logger.info("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    # Create loss criterion & optimizer
    # criterion = nn.CrossEntropyLoss()
    criterion = LabelSmoothing(vocab_size,0,smoothing=args.smoothing)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Start evaluation
    logger.info("Evaluation Started".center(60, '#'))
    for epoch in range(start_epoch, start_epoch+1):
        # Test the model
        wer = test(model, criterion, trainloader, device, epoch, logger, args.log_interval, writer, reverse_dict)

    logger.info("Evaluation Finished".center(60, '#'))

