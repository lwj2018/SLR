import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd
import spacy
import time
import json
import numpy
import sys
sys.path.append('/home/liweijie/projects/SLR')
from utils.textUtils import *
import torch.nn.functional as F

class Record:
    def __init__(self,path,sentence):
        self.skeleton_path = path
        self.sentence = sentence

"""
Implementation of CSL Phoenix Dataset
"""
class CSL_Phoenix_Openpose(Dataset):
    def __init__(self,skeleton_root='',annotation_file='',
                dictionary=None,
                clip_length=32,
                stride=4,
                upsample_rate=2,
                is_normalize=True,
                is_aug=True):
        super(CSL_Phoenix_Openpose,self).__init__()
        self.skeleton_root = skeleton_root
        self.annotation_file = annotation_file
        self.clip_length = clip_length
        self.stride = stride
        self.dictionary = dictionary
        self.stride = stride
        self.is_normalize = is_normalize
        self.is_aug = is_aug
        self.upsample_rate = upsample_rate
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
        sentence = sentence + ['<eos>']
        indices = []
        for word in sentence:
            if word in self.dictionary.keys():
                indices.append(self.dictionary[word])
            else:
                # the index of <unk> is 3
                indices.append(3)
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
        # Normalize
        if self.is_normalize:
            mat = self.normalize(mat)
        mat = torch.Tensor(mat)
        return mat


    def read_skeletons(self, frame_path, N):
        frame_path = os.path.join(self.skeleton_root,frame_path)

        skeleton_list = os.listdir(frame_path)
        skeleton_list.sort()
        # Ignore first frame which is blank
        skeleton_list = skeleton_list[1:]
        skeletons = []

        # Tatol number of clips must be larger than N
        # so length of skeletons must be larger the (N-1)* stride + clip_length
        l = len(skeleton_list)
        ur = self.upsample_rate
        remainder_num = (l-self.clip_length//ur)%(self.stride//ur)
        r = self.stride//ur - remainder_num if remainder_num>0 else 0
        clips_num = (l+r-self.clip_length//ur)//(self.stride//ur)
        if clips_num < N: r = r + (N-clips_num)*(self.stride//ur)
        # Read skeleton data
        for i in range(l):
            skeleton = self.read_json(os.path.join(frame_path, skeleton_list[i]))
            skeletons.append(skeleton)
        # Padding
        for i in range(r):
            skeletons.append(skeletons[-1])
        # After stack, shape is L x J x D, where L is perfectly suit for framing
        data = torch.stack(skeletons, dim=0)
        if self.is_aug:
            data = self.augmentation(data)
        # Upsample
        L,J,D = data.size()
        data = data.unsqueeze(0).permute(0,3,1,2)
        data = F.upsample(data,size=(self.upsample_rate*L,J),mode='bilinear').contiguous()
        data = data.permute(0,2,3,1).squeeze(0)
        # Resample the skeleton sequence with framing
        samples = []
        for i in range(0,(l+r)*ur-self.clip_length+1,self.stride):
            sample = data[i:i+self.clip_length]
            samples.append(sample)
        data = torch.cat(samples, dim=0)
        # After view, shape of data is S x clip_length x J x D, where S is sequence length
        data = data.view( (-1,self.clip_length) + data.size()[-2:] )
        return data

    def __getitem__(self, idx):
        record = self.data_list[idx]
        skeleton_path = record.skeleton_path
        sentence = record.sentence
        N = len(sentence)
        skeletons = self.read_skeletons(skeleton_path,N)
        sentence = torch.LongTensor(sentence)

        return {'input':skeletons, 'tgt':sentence}
        
    def __len__(self):
        return len(self.data_list)

    def normalize(self,mat):
        # Shape of mat is: J x D
        max_x = numpy.max(mat[:,0])
        min_x = min(mat[:,0])
        max_y = numpy.max(mat[:,1])
        min_y = min(mat[:,1])
        center_x = (max_x+min_x)/2
        center_y = (max_y+min_y)/2
        mat = (mat-[center_x,center_y])/[(max_x-min_x)/2,(max_y-min_y)/2]
        # TEST
        # print("max_x: %.2f,min_x: %.2f,max_y: %.2f,min_y: %.2f"%(max_x,min_x,max_y,min_y))
        return mat

    def augmentation(self,mat):
        amp = 0.02
        d0,d1,d2 = mat.size()
        jitter = (torch.rand(d0,d1,d2)-0.5)*amp*2
        mat = mat + jitter
        return mat

def min(array):
    threshold = 0.1
    min = 999999
    for x in array:
        if x>threshold and x<min:
            min = x
    return min

# Test
if __name__ == '__main__':
    # Path settings
    train_skeleton_root = "/mnt/data/haodong/openpose_output/train"
    train_annotation_file = "/mnt/data/public/datasets/phoenix2014-release/phoenix-2014-signerindependent-SI5/annotations/manual/train.SI5.corpus.csv"
    dev_skeleton_root = "/mnt/data/haodong/openpose_output/dev"
    dev_annotation_file = "/mnt/data/public/datasets/phoenix2014-release/phoenix-2014-signerindependent-SI5/annotations/manual/dev.SI5.corpus.csv"
    # Build dictionary
    dictionary = build_dictionary([train_annotation_file,dev_annotation_file])
    dataset = CSL_Phoenix_Openpose(skeleton_root=train_skeleton_root,annotation_file=train_annotation_file,dictionary=dictionary)
    print(dataset[3000]['input'].size())


