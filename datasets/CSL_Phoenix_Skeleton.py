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
    def __init__(self,skeleton_root='',annotation_file='',transform=None,dictionary=None,
                clip_length=16,
                stride=4):
        super(CSL_Phoenix_Skeleton,self).__init__()
        self.skeleton_root = skeleton_root
        self.annotation_file = annotation_file
        self.transform = transform
        self.clip_length = clip_length
        self.stride = stride
        self.dictionary = dictionary
        self.stride = stride
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
        # 第3维是置信度，不需要
        mat = mat[:,:2]
        mat = torch.Tensor(mat)
        return mat

    def read_skeleton(self, skeleton_path, N):
        skeleton_path = os.path.join(self.skeleton_root,skeleton_path)

        skeleton_list = os.listdir(skeleton_path)
        skeleton_list.sort()
        skeletons = []

        # Tatol number of clips must be larger than N
        # so length of skeletons must be larger the (N-1)* stride + clip_length
        l = len(skeleton_list)
        remainder_num = (l-self.clip_length)%self.stride
        r = self.stride - remainder_num if remainder_num>0 else 0
        clips_num = (l+r-self.clip_length)//self.stride
        if clips_num < N: r = r + (N-clips_num)*self.stride
        # Read skeleton data
        for i in range(l):
            skeleton = self.read_json(os.path.join(skeleton_path, skeleton_list[i]))
            skeletons.append(skeleton)
        # Padding
        for i in range(r):
            skeletons.append(skeletons[-1])
        # After stack, shape is L x J x D, where L is divisible by clip length
        skeletons = torch.stack(skeletons, dim=0)
        # Resample the skeleton sequence with sliding window
        samples = []
        for i in range(0,l+r-self.clip_length+1,self.stride):
            sample = skeletons[i:i+self.clip_length]
            samples.append(sample)
        data = torch.stack(samples, dim=0)
        # After view, shape of data is S x 16 x J x D, where S is sequence length
        data = data.view( (-1,self.clip_length) + data.size()[-2:] )
        return data

    def __getitem__(self, idx):
        record = self.data_list[idx]
        skeleton_path = record.skeleton_path
        sentence = record.sentence
        N = len(sentence)
        skeletons = self.read_skeleton(skeleton_path,N)
        sentence = torch.LongTensor(sentence)

        return {'input':skeletons, 'tgt':sentence}
        
    def __len__(self):
        return len(self.data_list)

        


