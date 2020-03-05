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


class CSL_Isolated_Multimodal(data.Dataset):
    
    def __init__(self, video_root, skleton_root, list_file,
                transform=None, length=32, image_length=16):
        self.video_root = video_root
        self.skeleton_root = skeleton_root
        self.list_file = list_file
        self.transform = transform
        self.length = length
        self.image_length = image_length
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
        num_frames = record.num_frames if record.num_frames<mat.shape[0]\
            else mat.shape[0]
        skeleton_indices,image_indices = self.get_sample_indices(num_frames)
                
        # Get images
        # the shape of images is C x H x W 
        # in which C is concat by RGB, then lrf, then length(16)
        video_path = osp.join(self.video_root,record.path)
        images = self.get_hand_and_face(video_path,mat,image_indices)
        images = self.transform(images)
    
        mat = mat[skeleton_indices,:,:]

        return mat, images, record.label
        

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
        skeleton_indices = indices
        image_indices = indices[::self.length//self.image_length]
        return skeleton_indices,image_indices
    
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

    def get_hand_and_face(self, video_path, skeleton, indices):
        # List all the files below this folder
        path_list = os.listdir(osp.join(self.video_root,video_path))
        path_list.sort()
        # Get hand & face
        parts_list = []
        scale = self.length//self.image_length
        for i,ind in enumerate(indices):
            image = Image.open(osp.join(self.video_root,path_list[ind]))
            # 根据openpose的关节标号,左手标号为=4，右手标号=7...
            lefthand_coord = skeleton[ind,4]
            righthand_coord = skeleton[ind,7]
            nose_coord = skeleton[ind,0]
            # 截取局部
            lefthand = self.crop_from_image(image,lefthand_coord,self.hand_cut_size)
            righthand = self.crop_from_image(image,righthand_coord,self.hand_cut_size)
            face = self.crop_from_image(image,nose_coord,self.face_cut_size)
            # 采集
            parts = [lefthand, righthand, face]
            parts_list.extend(parts)
        return parts_list

    def crop_from_image(self, image, coord, size):
        x, y =  coord
        xs = clip(x - size//2, self.width-1)
        xe = clip(x + size//2, self.width-1)
        ys = clip(y - size//2, self.height-1)
        ye = clip(y + size//2, self.height-1)
        img = np.array(image)
        img = img[ys:ye,xs:xe]
        image = Image.fromarray(img)
        image = image.resize((size,size),Image.BILINEAR)
        return image

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


def clip(x, max):
    if x < 0:
        x = 0
    if x > max:
        x = max
    return x