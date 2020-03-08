f = open('dictionary.txt','r')
words = f.readlines()

dictionary = {}
for word in words:
    data = word.rstrip('\n').split()
    index = int(data[0])
    token = data[1]
    dictionary[token] = index

import jieba
f = open('corpus.txt','r')
sentences = f.readlines()
count = 0
csl_dictionary = {}
punctuation = [' ','\n','\ufeff']
for sentence in sentences:
    words = jieba.cut(sentence.rstrip('\n'))
    for word in words:
        if word not in csl_dictionary.values() and word not in punctuation and '0' not in word:
            csl_dictionary[count] = word
            count += 1

def isIndict(k,dictionary):
    for word in dictionary.keys():
        if k in word:
            return word
    return -1
isl_in_csl_dictionary = {}
for k in csl_dictionary.values():
    word = isIndict(k,dictionary)
    if word!=-1:
        index = dictionary[word]
        isl_in_csl_dictionary[k] = index
isl_in_csl_dictionary = sorted(isl_in_csl_dictionary.items(),key=lambda item:item[1])

# Generate subset file for validation
subset_index_list = [record[1] for record in isl_in_csl_dictionary]

import os
import os.path as osp

def create_path(path):
    if not osp.exists(path):
        os.makedirs(path)

num_class = 500
color_video_root = "/home/liweijie/SLR_dataset/S500_color_video"
skeleton_root = "/home/liweijie/SLR_dataset/xf500_body_color_txt"
val_list = open("../input/subset_val_list.txt","w")

color_video_path_list = os.listdir(color_video_root)
color_video_path_list.sort()
n = len(color_video_path_list)
for i,color_video_path in enumerate(color_video_path_list):
    print("%d/%d"%(i,n))
    label = color_video_path
    abs_color_video_path = osp.join(color_video_root,color_video_path)
    color_video_list = os.listdir(abs_color_video_path)
    color_video_list.sort()
    index = int(label)
    if index in subset_index_list:
        for color_video in color_video_list:
            abs_color_video = osp.join(abs_color_video_path,color_video)
            if(osp.isdir(abs_color_video)):
                p = color_video.split('_')
                person = int(p[0].lstrip('P'))
                num_frames = len(os.listdir(abs_color_video))
                path = osp.join(color_video_path,color_video)
                if not '(' in path:
                    path_skeleton = path.rstrip("color")+"body.txt"
                    abs_path_skeleton = osp.join(skeleton_root,path_skeleton)
                    if osp.exists(abs_path_skeleton):
                        record = path+"\t"+path_skeleton+"\t"+\
                                            str(num_frames)+"\t"+color_video_path+"\n"
                        val_list.write(record)
                    else:
                        print("The skeleton path %s don't exist"%abs_path_skeleton)