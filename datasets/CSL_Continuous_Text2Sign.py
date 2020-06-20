import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd
import spacy
import time
import jieba
import json
import sys
import numpy
sys.path.append('/home/liweijie/projects/SLR')
from utils.textUtils import *
from torch.nn import functional as F

class Record:
    def __init__(self,data):
        data = data.rstrip('\n').split()
        self.frame_path = data[0]
        self.sentence = data[1]

"""
Implementation of CSL Phoenix Dataset
"""
class CSL_Continuous_Text2Sign(Dataset):
    def __init__(self,
        setname,
        skeleton_root='',
        list_file='',
        dictionary=None,
        upsample_rate=2,
        add_two_end=False,
        is_normalize=True):
        super(CSL_Continuous_Text2Sign,self).__init__()
        self.setname = setname
        self.skeleton_root = skeleton_root
        self.list_file = list_file
        self.dictionary = dictionary
        self.upsample_rate = upsample_rate
        self.add_two_end = add_two_end
        self.is_normalize = is_normalize

        self.get_data_list()
    
    def get_data_list(self):
        f = open(self.list_file,'r')
        self.data_list = []
        for data in f.readlines():
            record = Record(data)
            pind = record.frame_path.find('P')
            person = int(record.frame_path[pind+1:pind+3])
            label = int(record.frame_path.split('/')[0])
            if(person==1):
                if self.setname=='train' and label<80:
                    self.data_list.append(record)
                elif self.setname=='test' and label>=80:
                    self.data_list.append(record)

    def read_json(self,jsonFile):
        skeletonDict = json.load(open(jsonFile,'rb'))
        bodySkeleton = numpy.array(skeletonDict['Body']).squeeze()
        leftHandSkeleton = numpy.array(skeletonDict['Left hand']).squeeze()
        rightHandSkeleton = numpy.array(skeletonDict['Right hand']).squeeze()
        mat = numpy.concatenate([bodySkeleton,leftHandSkeleton,rightHandSkeleton],-2)
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

        # Read skeleton data
        l = len(skeleton_list)
        for i in range(l):
            skeleton = self.read_json(os.path.join(frame_path, skeleton_list[i]))
            skeletons.append(skeleton)
        # After stack, shape is L x J x D
        data = torch.stack(skeletons, dim=0)
        # Upsample
        L,J,D = data.size()
        data = data.unsqueeze(0).permute(0,3,1,2)
        data = F.upsample(data,size=(self.upsample_rate*L,J),mode='bilinear').contiguous()
        data = data.permute(0,2,3,1).squeeze(0)
        return data

    def __getitem__(self, idx):
        record = self.data_list[idx]
        frame_path = record.frame_path
        sentence = convert_chinese_to_indices(record.sentence,self.dictionary,self.add_two_end) 
        N = len(sentence)
        skeletons = self.read_skeletons(frame_path,N)
        sentence = torch.LongTensor(sentence)

        return {'input':sentence, 'tgt':skeletons}
        
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

def min(array):
    threshold = 0.1
    min = 999999
    for x in array:
        if x>threshold and x<min:
            min = x
    return min

# Test
if __name__ == '__main__':
    # Build dictionary
    dictionary = build_dictionary_for_t2s()
    # Path settings
    skeleton_root = "/home/haodong/Data/CSL_Continuous_Skeleton"
    train_list = "/home/liweijie/Data/public_dataset/train_list.txt"
    val_list = "/home/liweijie/Data/public_dataset/val_list.txt"
    dataset = CSL_Continious_Text2Sign('train',skeleton_root=skeleton_root,list_file=train_list,dictionary=dictionary)
    print(len(dataset))
    print(dataset[10]['input'].size(),dataset[10]['tgt'].size())


