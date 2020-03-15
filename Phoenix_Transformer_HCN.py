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
from datasets.CSL_Phoenix_Openpose import CSL_Phoenix_Openpose
from args import Arguments
from utils.ioUtils import save_checkpoint, resume_model
from utils.critUtils import LabelSmoothing
from utils.textUtils import build_dictionary, reverse_dictionary
from torch.utils.tensorboard import SummaryWriter
import warnings
 
warnings.filterwarnings('ignore')

# Path setting
train_skeleton_root = "/mnt/data/haodong/openpose_output/train"
train_annotation_file = "/mnt/data/public/datasets/phoenix2014-release/phoenix-2014-signerindependent-SI5/annotations/manual/train.SI5.corpus.csv"
dev_skeleton_root = "/mnt/data/haodong/openpose_output/dev"
dev_annotation_file = "/mnt/data/public/datasets/phoenix2014-release/phoenix-2014-signerindependent-SI5/annotations/manual/dev.SI5.corpus.csv"
# Hyper params
learning_rate = 1e-6
batch_size = 4
epochs = 1000
num_classes = 512
sample_size = 128
clip_length = 16
smoothing = 0.1
stride = 8
# Options
store_name = 'Transformer_HCN'
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

def collate(batch):
    sources = []
    targets = []
    src_len_list = []
    tgt_len_list = []
    max_src_len = 0
    max_tgt_len = 0
    # shape of source is: s x 16 x J x D
    # shape of target is: t
    for sample in batch:
        source = torch.Tensor(sample['input'])
        target = torch.LongTensor(sample['tgt'])
        # remember max src len
        src_len = source.size(0)
        if src_len > max_src_len:
            max_src_len = src_len
        # remember max tg len
        tgt_len = target.size(0)
        if tgt_len > max_tgt_len:
            max_tgt_len = tgt_len
        # append
        src_len_list.append(src_len)
        tgt_len_list.append(tgt_len)
        sources.append(source)
        targets.append(target)
    # fill zeros
    full_sources = []
    full_targets = []
    for source, target, src_len, tgt_len in zip(sources, targets, src_len_list, tgt_len_list):
        # pad the source
        src_pad = torch.zeros( (max_src_len-src_len,) + source.size()[-3:] )
        full_src = torch.cat([source, src_pad], 0)
        full_sources.append(full_src)
        # pad the target
        tgt_pad = torch.zeros(max_tgt_len-tgt_len,dtype=torch.long)
        full_tgt = torch.cat([target,tgt_pad], 0)
        full_targets.append(full_tgt)
    # After data processing,
    # shape of src is N x S x 16 x J x D, where S is max length of this batch
    # shape of tgt is N x T, where T is max length of this batch
    # shape of src_len_list is N
    # shape of tgt_len_list is N
    src = torch.stack(full_sources, 0)
    tgt = torch.stack(full_targets, 0)
    src_len_list = torch.LongTensor(src_len_list)
    tgt_len_list = torch.LongTensor(tgt_len_list)
    return {'src':src, 'tgt':tgt, 'src_len_list':src_len_list, 'tgt_len_list':tgt_len_list}

# Train with Transformer
if __name__ == '__main__':
    # build dictionary
    dictionary = build_dictionary([train_annotation_file,dev_annotation_file])
    reverse_dict = reverse_dictionary(dictionary)
    vocab_size = len(dictionary)
    print("The size of vocabulary is %d"%vocab_size)
    # Load data
    trainset = CSL_Phoenix_Openpose(skeleton_root=train_skeleton_root,annotation_file=train_annotation_file,
            dictionary=dictionary,clip_length=clip_length,stride=stride)
    devset = CSL_Phoenix_Openpose(skeleton_root=dev_skeleton_root,annotation_file=dev_annotation_file,
            dictionary=dictionary,clip_length=clip_length,stride=stride)
    print("Dataset samples: {}".format(len(trainset)+len(devset)))
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True,
            collate_fn=collate)
    testloader = DataLoader(devset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True,
            collate_fn=collate)
    # Create model
    model = CSL_Transformer(vocab_size,vocab_size,sample_size=sample_size, clip_length=clip_length,
                num_classes=num_classes,modal='skeleton').to(device)
    if checkpoint is not None:
        start_epoch, best_wer = resume_model(model,checkpoint)
    # Run the model parallelly
    if torch.cuda.device_count() > 1:
        print("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    # Create loss criterion & optimizer
    # criterion = nn.CrossEntropyLoss(ignore_index=0)
    criterion = LabelSmoothing(vocab_size,0,smoothing=smoothing)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Start training
    print("Training Started".center(60, '#'))
    for epoch in range(start_epoch, epochs):
        # Test the model
        wer = test(model, criterion, testloader, device, epoch, log_interval, writer, reverse_dict)
        # Train the model
        train(model, criterion, optimizer, trainloader, device, epoch, log_interval, writer, reverse_dict)
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



