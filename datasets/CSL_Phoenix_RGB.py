import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd
import spacy
import time
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/liweijie/projects/SLR')
from utils.textUtils import *

class Record:
    def __init__(self,path,sentence):
        self.frame_path = path
        self.sentence = sentence

"""
Implementation of CSL Phoenix Dataset
"""
class CSL_Phoenix_RGB(Dataset):
    def __init__(self,frame_root='',annotation_file='',transform=None,dictionary=None,
        clip_length=16,
        stride=8):
        super(CSL_Phoenix_RGB,self).__init__()
        self.frame_root = frame_root
        self.annotation_file = annotation_file
        self.transform = transform
        self.clip_length = clip_length
        self.stride = stride
        self.dictionary = dictionary
        self.prepare()
        self.get_data_list()

    def prepare(self):
        # Prepare df, lang_model et. al.
        df = pd.read_csv(self.annotation_file,sep='|')
        lang_model = spacy.load('de')
        punctuation = ['_','NULL','ON','OFF','EMOTION','LEFTHAND','IX','PU']

        self.punctuation = punctuation
        self.lang_model = lang_model
        self.df = df

    def process_sentence(self,sentence):
        # Tokenize & convert tokens to indices
        sentence = [tok.text for tok in self.lang_model.tokenizer(sentence) 
            if not tok.text in self.punctuation]
        sentence = ['<bos>'] + sentence + ['<eos>']
        indices = [self.dictionary[word] for word in sentence 
            if word in self.dictionary.keys()]
        return indices
    
    def get_data_list(self):
        # Prepare data list
        self.data_list = []
        for i in range(len(self.df)):
            row = self.df.loc[i]
            frame_path = row['id']
            sentence = row['annotation']
            sentence = self.process_sentence(sentence)
            record = Record(frame_path,sentence)
            self.data_list.append(record)
        # Temporarily setting
        # self.data_list = self.data_list[:100]

    def read_images(self, frame_path, N):
        # 由于phoenix神奇的数据集文件结构
        frame_path = os.path.join(self.frame_root,frame_path) + "/1/"

        imagename_list = os.listdir(frame_path)
        imagename_list.sort()
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
        # Preview for test
        # self.preview(data)
        return data

    def __getitem__(self, idx):
        record = self.data_list[idx]
        frame_path = record.frame_path
        sentence = record.sentence
        N = len(sentence)
        images = self.read_images(frame_path,N)
        sentence = torch.LongTensor(sentence)

        return {'input':images, 'tgt':sentence}
        
    def __len__(self):
        return len(self.data_list)

    def preview(self,data):
        # After permute, shape resume to S x 16 x H x W x C
        data = data.permute(0, 2, 3, 4, 1)
        # Select first clip 
        data = data[0]
        for i,frame in enumerate(data):
            plt.subplot(4,4,i+1)
            plt.imshow(frame)
        plt.show()
        
# Test
if __name__ == '__main__':
    sample_size = 128
    # get transform
    transform = transforms.Compose([transforms.Resize([sample_size, sample_size]),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5], std=[0.5])])
    # Path settings
    train_frame_root = "/mnt/data/public/datasets/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px/train"
    train_annotation_file = "/mnt/data/public/datasets/phoenix2014-release/phoenix-2014-multisigner/annotations/manual/train.corpus.csv"
    dev_annotation_file = "/mnt/data/public/datasets/phoenix2014-release/phoenix-2014-multisigner/annotations/manual/dev.corpus.csv"
    # build dictionary
    dictionary = build_dictionary([train_annotation_file,dev_annotation_file])
    # build dataset
    dataset = CSL_Phoenix_RGB(frame_root=train_frame_root,annotation_file=train_annotation_file,transform=transform,
                dictionary=dictionary)
    dataset[1000]

