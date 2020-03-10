import torch.utils.data as data

from PIL import Image
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

import os
import os.path as osp
import time
import matplotlib.pyplot as plt

# Params
width = 1280
height = 720
face_cut_size = 100
hand_cut_size = 150

class VideoRecord(object):
    def __init__(self,row):
        self._data = row
    @property
    def path(self):
        return self._data[0]
    @property
    def skeleton_path(self):
        return self._data[1]
    @property
    def num_frames(self):
        return int(self._data[2])
    @property
    def label(self):
        return int(self._data[3])


class CSL_Isolated_Openpose(data.Dataset):
    
    def __init__(self, skeleton_root, list_file, length=32):
        self.skeleton_root = skeleton_root
        self.list_file = list_file
        self.length = length
        self.width = width
        self.height = height
        self.face_cut_size = face_cut_size
        self.hand_cut_size = hand_cut_size
        
        self._parse_list()

    def __getitem__(self, index):
        record = self.video_list[index]
        # Get mat
        # The shape of mat is T x J x D
        mat = self._load_data(record.skeleton_path)
        mat = self.select_skeleton_indices(mat)
        num_frames = record.num_frames if record.num_frames<mat.shape[0]\
            else mat.shape[0]
        indices = self.get_sample_indices(num_frames)    
        mat = mat[indices,:,:]

        return mat, record.label
        

    def __len__(self):
        return len(self.video_list)
        
    def _parse_list(self):
        tmp = [x.strip().split('\t') for x in open(self.list_file)]
        tmp = [item for item in tmp if int(item[2])>4]
        self.video_list = [VideoRecord(item) for item in tmp]
        print('video number:%d'%(len(self.video_list)))

    def get_sample_indices(self,num_frames):
        indices = np.linspace(1,num_frames-1,self.length).astype(int)
        interval = (num_frames-1)//self.length
        if interval>0:
            jitter = np.random.randint(0,interval,self.length)
        else:
            jitter = 0
        jitter = (np.random.rand(self.length)*interval).astype(int)
        indices = np.sort(indices+jitter)
        indices = np.clip(indices,0,num_frames-1)
        return indices
    
    def _load_data(self, path):
        # 对于openpose数据集要处理训练列表
        path = path.rstrip("body.txt")+"color"
        path = osp.join(self.skeleton_root, path)
        file_list = os.listdir(path)
        file_list.sort()
        mat = []
        for i,file in enumerate(file_list):
            # 第一帧有问题，先排除
            if i>0:
                filename =  osp.join(path,file)
                f = open(filename,"r")
                content = f.readlines()
                try:
                    mat_i = self.content_to_mat(content)
                    mat.append(mat_i)
                except:
                    print("can not convert this file to mat: "+filename)
        mat = np.array(mat)
        end = time.time()
        mat = mat.astype(np.float32)
        return mat


    def content_to_mat(self,content):
        mat = []
        for i in range(len(content)):
            if "Body" in content[i]:
                for j in range(25):
                    record = content[i+1+j].lstrip().lstrip("[").rstrip("\n").rstrip("]")
                    joint = [float(x) for x in record.split()]
                    mat.append(joint)
            elif "Face" in content[i]:
                for j in range(70):
                    record = content[i+1+j].lstrip().lstrip("[").rstrip("\n").rstrip("]")
                    joint = [float(x) for x in record.split()]
                    mat.append(joint)

            elif "Left" in content[i]:
                for j in range(21):
                    record = content[i+1+j].lstrip().lstrip("[").rstrip("\n").rstrip("]")
                    joint = [float(x) for x in record.split()]
                    mat.append(joint)

            elif "Right" in content[i]:
                for j in range(21):
                    record = content[i+1+j].lstrip().lstrip("[").rstrip("\n").rstrip("]")
                    joint = [float(x) for x in record.split()]
                    mat.append(joint)
                break

        mat = np.array(mat)
        # 第三维是置信度，不需要
        mat = mat[:,0:2]
        return mat

    def select_skeleton_indices(self,input):
        left_arm = input[:,[3,4],:]
        right_arm = input[:,[6,7],:]
        face = input[:,25:95,:]
        left_hand = input[:,95:116,:]
        right_hand = input[:,116:137,:]
        x = np.concatenate([left_arm,right_arm,face,\
            left_hand,right_hand],1)
        return x


