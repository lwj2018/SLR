import os
import sys
from datetime import datetime
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from models.Transformer import CSL_Transformer

from datasets.CSL_Phoenix import CSL_Phoenix, build_dictionary, reverse_dictionary
from args import Arguments
from utils.ioUtils import resume_model
from utils.evalUtils import eval

# get arguments
args = Arguments()

# Log to file
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.FileHandler(args.eval_log_path), logging.StreamHandler()])
logger = logging.getLogger('SLR')
logger.info('Logging to file...')

# Use specific gpus
os.environ["CUDA_VISIBLE_DEVICES"]=args.device_list
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# qualitative evaluation with Transformer
if __name__ == '__main__':

    # build dictionary
    dictionary = build_dictionary([args.train_annotation_file,args.dev_annotation_file])
    vocab_size = len(dictionary)
    print("The size of vocabulary is %d"%vocab_size)

    # reverse the dictionary for itos convertion
    reverse_dict = reverse_dictionary(dictionary)

    # Load data
    transform = transforms.Compose([transforms.Resize([args.sample_size, args.sample_size]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])
    trainset = CSL_Phoenix(frame_root=args.train_frame_root,annotation_file=args.train_annotation_file,
        transform=transform,dictionary=dictionary)
    devset = CSL_Phoenix(frame_root=args.dev_frame_root,annotation_file=args.dev_annotation_file,
        transform=transform,dictionary=dictionary)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
    testloader = DataLoader(devset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)

    # Create model
    model = CSL_Transformer(vocab_size,vocab_size,sample_size=args.sample_size, clip_length=args.clip_length,
                num_classes=args.num_classes).to(device)
    # Resume from checkpoint
    resume_model(model,args.eval_checkpoint)
                
    # Run the model parallelly
    if torch.cuda.device_count() > 1:
        logger.info("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    # Start evaluation
    logger.info("Qualitative Evaluation".center(60, '#'))
    for epoch in range(args.epochs):
        eval(model,epoch,testloader,device,logger,reverse_dict)

