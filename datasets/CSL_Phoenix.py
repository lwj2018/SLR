import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd
import spacy
import time
from utils.textUtils import build_dictionary, reverse_dictionary

class Record:
    def __init__(self,path,sentence):
        self.frame_path = path
        self.sentence = sentence

"""
Implementation of CSL Phoenix Dataset
"""
class CSL_Phoenix(Dataset):
    def __init__(self,frame_root='',annotation_file='',transform=None,dictionary=None):
        super(CSL_Phoenix,self).__init__()
        self.frame_root = frame_root
        self.annotation_file = annotation_file
        self.transform = transform
        self.clip_length = 16
        self.dictionary = dictionary
        self.prepare()
        self.get_data_list()

    def prepare(self):
        df = pd.read_csv(self.annotation_file,sep='|')
        lang_model = spacy.load('de')
        punctuation = ['_','NULL','ON','OFF','EMOTION','LEFTHAND','IX','PU']

        self.punctuation = punctuation
        self.lang_model = lang_model
        self.df = df

    def process_sentence(self,sentence):
        sentence = [tok.text for tok in self.lang_model.tokenizer(sentence) 
            if not tok.text in self.punctuation]
        sentence = ['<bos>'] + sentence + ['<eos>']
        indices = [self.dictionary[word] for word in sentence 
            if word in self.dictionary.keys()]
        return indices
    
    def get_data_list(self):
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

    def read_images(self, frame_path):
        # 由于phoenix神奇的数据集文件结构
        frame_path = os.path.join(self.frame_root,frame_path) + "/1/"

        imagename_list = os.listdir(frame_path)
        imagename_list.sort()
        # assert len(imagename_list) >= self.clip_length, \
        #     "Too few images in your data folder: " + str(frame_path)
        images = []

        remainder_num = len(imagename_list)%self.clip_length
        # clip
        if remainder_num >= self.clip_length/2 or len(imagename_list)<self.clip_length:
            l = len(imagename_list)
        elif remainder_num < self.clip_length/2:
            l = len(imagename_list)//self.clip_length * self.clip_length
        for i in range(l):
            image = Image.open(os.path.join(frame_path, imagename_list[i])).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            images.append(image)
        # padding
        if remainder_num >= self.clip_length/2 or len(imagename_list)<self.clip_length:
            for i in range(self.clip_length-remainder_num):
                images.append(images[-1])

        images = torch.stack(images, dim=0)
        # switch dimension for 3d cnn
        # shape of C x L x H x W
        images = images.permute(1, 0, 2, 3)
        # print(images.shape)
        return images

    def __getitem__(self, idx):
        record = self.data_list[idx]
        frame_path = record.frame_path
        sentence = record.sentence
        images = self.read_images(frame_path)
        sentence = torch.LongTensor(sentence)

        return {'input':images, 'tgt':sentence}
        
    def __len__(self):
        return len(self.data_list)

        


