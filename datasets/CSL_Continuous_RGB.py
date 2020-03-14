import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd
import spacy
import time
import jieba
from utils.textUtils import *

class Record:
    def __init__(self,data):
        data = data.rstrip('\n').split()
        self.frame_path = data[0]
        self.sentence = data[1]

"""
Implementation of CSL Phoenix Dataset
"""
class CSL_Continuous_RGB(Dataset):
    def __init__(self,frame_root='',list_file='',
        transform=None,
        dictionary=None,
        clip_length=16,
        stride=8):
        super(CSL_Continuous_RGB,self).__init__()
        self.frame_root = frame_root
        self.transform = transform
        self.clip_length = clip_length
        self.stride = stride
        self.dictionary = dictionary
        self.list_file = list_file
        self.get_data_list()
    
    def get_data_list(self):
        f = open(self.list_file,'r')
        self.data_list = [Record(data) for data in f.readlines()]

    def read_images(self, frame_path, N):
        frame_path = os.path.join(self.frame_root,frame_path)

        imagename_list = os.listdir(frame_path)
        imagename_list.sort()
        # Down sample by four times
        imagename_list = imagename_list[::4]
        images = []

        # Tatol number of clips must be larger than N
        # so length of images must be larger the (N-1)* stride + clip_length
        l = len(imagename_list)
        remainder_num = (l-self.clip_length)%self.stride
        r = self.stride - remainder_num if remainder_num>0 else 0
        clips_num = (l+r-self.clip_length)//self.stride
        if clips_num < N: r = r + (N-clips_num)*self.stride
        # Read image data
        for i in range(l):
            image = Image.open(os.path.join(frame_path, imagename_list[i])).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            images.append(image)
        # Padding
        for i in range(r):
            images.append(images[-1])
        # After stack, shape is L x C x H x W, where L is divisible by clip length
        images = torch.stack(images, dim=0)
        # Resample the image sequence with sliding window
        samples = []
        for i in range(0,l+r-self.clip_length+1,self.stride):
            sample = images[i:i+self.clip_length]
            samples.append(sample)
        data = torch.stack(samples, dim=0)
        # After view, shape of data is S x 16 x C x H x W
        data = data.view( (-1,self.clip_length) + data.size()[-3:] )
        # After permute, shape of data is S x C x 16 x H x W
        data = data.permute(0, 2, 1, 3, 4)
        return data

    def __getitem__(self, idx):
        record = self.data_list[idx]
        frame_path = record.frame_path
        sentence = convert_chinese_to_indices(record.sentence,self.dictionary) 
        N = len(sentence)
        images = self.read_images(frame_path,N)
        sentence = torch.LongTensor(sentence)

        return {'input':images, 'tgt':sentence}
        
    def __len__(self):
        return len(self.data_list)

        


