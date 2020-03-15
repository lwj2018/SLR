import pandas as pd
import spacy
from collections import Counter
import time
import jieba

def build_dictionary(files):
    start = time.time()
    lang_model = spacy.load('de')
    end = time.time()
    print('load lang_model cost %.3f s'%(end-start))
    train = []
    # 合并annotation中的语料
    for file in files:
        df = pd.read_csv(file,sep='|')
        for i in range(len(df)):
            train.append(df.loc[i]['annotation'])

    # Create a dictionary which maps tokens to indices (train contains all the training sentences)
    freq_list = Counter()
    punctuation = ['_','NULL','ON','OFF','EMOTION','LEFTHAND','IX','PU']
    for sentence in train:
        sentence = [tok.text for tok in lang_model.tokenizer(sentence) if not tok.text in punctuation]
        freq_list.update(sentence)

    # 按照词的出现频率建立词典，词频越高索引越靠前
    freq_list = sorted(freq_list.items(),key=lambda item:item[1],reverse=True)
    dictionary = {}
    dictionary['<pad>'] = 0
    dictionary['<bos>'] = 1
    dictionary['<eos>'] = 2
    for i,item in enumerate(freq_list):
        dictionary[item[0]] = i+3
    print("Build dictionary successfully!")
    return dictionary

def reverse_dictionary(dictionary):
    reverse_dict = {}
    for k,v in dictionary.items():
        reverse_dict[v] = k
    return reverse_dict

def itos(idx_list, reverse_dict):
    sentence = [reverse_dict[idx] for idx in idx_list]
    return sentence

def stoi(token_list, dictionary):
    index_list = []
    for token in token_list:
        if token in dictionary.keys():
            index = dictionary[token]
            index_list.append(index)
    return index_list

def itos_clip(idx_list, reverse_dict):
    sentence = []
    for idx in idx_list:
        word = reverse_dict[idx]
        sentence.append(word)
        if word=='<eos>':
            break
    return sentence

def convert_chinese_to_indices(sentence, dictionary):
    words = jieba.cut(sentence.rstrip('\n'))
    words = ['<bos>'] + list(words) + ['<eos>']
    indices = stoi(words,dictionary)
    return indices

def build_isl_dictionary():
    dict_fname = '/home/liweijie/Data/SLR_dataset/dictionary.txt'
    f = open(dict_fname,encoding='utf-8')
    words = f.readlines()
    dictionary = {}
    for word in words:
        data = word.rstrip('\n').split()
        index = int(data[0])
        token = data[1]
        dictionary[token] = index+3
    dictionary[0] = '<pad>'
    dictionary[1] = '<bos>'
    dictionary[2] = '<eos>'
    return dictionary