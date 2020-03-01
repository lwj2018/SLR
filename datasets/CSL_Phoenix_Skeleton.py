import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd
import spacy
import time
from utils.textUtils import build_dictionary, reverse_dictionary
import json
import numpy

class Record:
    def __init__(self,path,sentence):
        self.skeleton_path = path
        self.sentence = sentence

"""
Implementation of CSL Phoenix Dataset
"""
class CSL_Phoenix_Skeleton(Dataset):
    def __init__(self,skeleton_root='',annotation_file='',transform=None,dictionary=None):
        super(CSL_Phoenix_Skeleton,self).__init__()
        self.skeleton_root = skeleton_root
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
            skeleton_path = row['id']
            sentence = row['annotation']
            sentence = self.process_sentence(sentence)
            record = Record(skeleton_path,sentence)
            self.data_list.append(record)

    def read_json(self,jsonFile):
        skeletonDict = json.load(open(jsonFile,'rb'))
        bodySkeleton = numpy.array(skeletonDict['Body']).squeeze()
        faceSkeleton = numpy.array(skeletonDict['Face']).squeeze()
        leftHandSkeleton = numpy.array(skeletonDict['Left hand']).squeeze()
        rightHandSkeleton = numpy.array(skeletonDict['Right hand']).squeeze()
        mat = numpy.concatenate([bodySkeleton,faceSkeleton,leftHandSkeleton,rightHandSkeleton],-2)
        # 处理误识别出两个骨架的情况
        if len(mat.shape)>2:
            mat = mat[0]
        mat = mat[:,:2]
        mat = torch.Tensor(mat)
        return mat

    def read_skeleton(self, skeleton_path):
        skeleton_path = os.path.join(self.skeleton_root,skeleton_path)

        skeleton_list = os.listdir(skeleton_path)
        skeleton_list.sort()
        # assert len(skeleton_list) >= self.clip_length, \
        #     "Too few skeletons in your data folder: " + str(skeleton_path)
        skeletons = []

        remainder_num = len(skeleton_list)%self.clip_length
        # clip
        if remainder_num >= self.clip_length/2 or len(skeleton_list)<self.clip_length:
            l = len(skeleton_list)
        elif remainder_num < self.clip_length/2:
            l = len(skeleton_list)//self.clip_length * self.clip_length
        for i in range(l):
            skeleton = self.read_json(os.path.join(skeleton_path, skeleton_list[i]))
            skeletons.append(skeleton)
        # padding
        if remainder_num >= self.clip_length/2 or len(skeleton_list)<self.clip_length:
            for i in range(self.clip_length-remainder_num):
                skeletons.append(skeletons[-1])

        # after stack, shape is T x J x D, where T is divisible by clip length
        skeletons = torch.stack(skeletons, dim=0)
        return skeletons

    def __getitem__(self, idx):
        record = self.data_list[idx]
        skeleton_path = record.skeleton_path
        sentence = record.sentence
        skeletons = self.read_skeleton(skeleton_path)
        sentence = torch.LongTensor(sentence)

        return {'input':skeletons, 'tgt':sentence}
        
    def __len__(self):
        return len(self.data_list)

        


