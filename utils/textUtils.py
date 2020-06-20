import pandas as pd
import spacy
from collections import Counter
import time
import jieba
import copy

csl_dictionary_file = '/home/liweijie/Data/SLR_dataset/dictionary.txt'
csl_corpus_file = "/home/liweijie/Data/public_dataset/corpus.txt"

def build_dictionary(file):
    start = time.time()
    lang_model = spacy.load('de')
    end = time.time()
    print('load lang_model cost %.3f s'%(end-start))
    train = []
    # 合并annotation中的语料
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
    dictionary['<unk>'] = 3
    count = 0
    for i,item in enumerate(freq_list):
        if item[1] > 2:
            dictionary[item[0]] = count+4
            count += 1
        else:
            dictionary[item[0]] = 3
    print("Build dictionary successfully!")
    return dictionary

def reverse_dictionary(dictionary):
    reverse_dict = {}
    for k,v in dictionary.items():
        reverse_dict[v] = k
    return reverse_dict

def reverse_phoenix_dictionary(dictionary):
    reverse_dict = {}
    for k,v in dictionary.items():
        # <unk> is a special case
        if v!= 3:
            reverse_dict[v] = k
        else:
            reverse_dict[v] = '<unk>'
    return reverse_dict

def itos(idx_list, reverse_dict):
    # ignore pad
    sentence = [reverse_dict[idx] for idx in idx_list if idx!=0]
    return sentence

def stoi(token_list, dictionary):
    index_list = []
    for token in token_list:
        if token in dictionary.keys():
            index = dictionary[token]
            index_list.append(index)
    return index_list

def compress(input):
    if len(input) == 0:
        return input
    input = list(input)
    last_item = input[0]
    compress_list = [last_item]
    for i,item in enumerate(input):
        if i>0:
            if item!=last_item:
                compress_list.append(item)
                last_item = item
    return compress_list

def itos_clip(idx_list, reverse_dict):
    sentence = []
    for idx in idx_list:
        # ignore pad
        if idx!=0:
            word = reverse_dict[idx]
            sentence.append(word)
            if word=='<eos>':
                break
    return sentence

def convert_chinese_to_indices(sentence, dictionary, add_two_end):
    words = []
    for word in jieba.cut(sentence.rstrip('\n')):
        word = manual_cut(word)
        words.extend(word)
    if add_two_end:
        words = ['<bos>'] + list(words) + ['<eos>']
    else:
        words = list(words)+['<eos>']
    indices = stoi(words,dictionary)
    return indices

def build_isl_dictionary():
    dict_fname = csl_dictionary_file
    f = open(dict_fname,encoding='utf-8')
    words = f.readlines()
    dictionary = {}
    for word in words:
        data = word.rstrip('\n').split()
        index = int(data[0])
        token = data[1]
        dictionary[token] = index
    return dictionary

def build_csl_dictionary():
    annotation_file = open(csl_corpus_file,'r')
    corpus = annotation_file.readlines()
    corpus = [sentence.rstrip('\n').split()[1] for sentence in corpus]
    # Create a dictionary which maps tokens to indices (train contains all the training sentences)
    freq_list = Counter()
    punctuation = ['\ufeff']
    for sentence in corpus:
        sentence = [word for word in jieba.cut(sentence) if not word in punctuation]
        freq_list.update(sentence)

    # 按照词的出现频率建立词典，词频越高索引越靠前
    freq_list = sorted(freq_list.items(),key=lambda item:item[1],reverse=True)
    dictionary = {}
    dictionary['<pad>'] = 0
    dictionary['<bos>'] = 1
    dictionary['<eos>'] = 2
    for i,item in enumerate(freq_list):
        dictionary[item[0]] = i+3
    print("Build CSL dictionary successfully!")
    return dictionary

def manual_cut(word):
    if word == '女朋友':
        return ['女','朋友']
    elif word == '现实情况':
        return ['现实','情况']
    elif word == '自由恋爱':
        return ['自由','恋爱']
    elif word == '扭转局面':
        return ['扭转','局面']
    elif word == '事业成功':
        return ['事业','成功']
    elif word == '经验丰富':
        return ['经验','丰富']
    elif word == '有雨':
        return ['有','雨']
    elif word == '他人':
        return ['别人']
    elif word == '圆满成功':
        return ['圆满','成功']
    elif word == '针线':
        return ['针','线']
    elif word == '星星':
        return ['星']
    elif word == '小孩子':
        return ['小孩儿（儿童、少年）']
    else:
        return [word]

def isIndict(k):
    global isl_dictionary
    for word in isl_dictionary.keys():
        if k == word:
            return word
        elif k in word and k!='的' and k!='有':
            isl_dictionary[k] = isl_dictionary[word]
            return word
    return -1

def build_dictionary_for_t2s():
    annotation_file = open(csl_corpus_file,'r')
    corpus = annotation_file.readlines()
    corpus = [sentence.rstrip('\n').split()[1] for sentence in corpus]
    # Create a dictionary which maps tokens to indices (train contains all the training sentences)
    freq_list = Counter()
    punctuation = ['\ufeff']
    for sentence in corpus:
        tmp_sentence = []
        for word in jieba.cut(sentence):
            if not word in punctuation:
                word = manual_cut(word)
                tmp_sentence.extend(word)
        freq_list.update(tmp_sentence)

    # 按照词的出现频率建立词典，词频越高索引越靠前
    freq_list = sorted(freq_list.items(),key=lambda item:item[1],reverse=True)
    csl_dictionary = {}
    for i,item in enumerate(freq_list):
        csl_dictionary[item[0]] = i

    # build isl dictionary
    isl_dictionary = build_isl_dictionary()

    # Update isl_dictionary to handle the words with same meanings
    tmp_dict = copy.deepcopy(isl_dictionary)
    for k in csl_dictionary.keys():
        for word in tmp_dict.keys():
            if k in word and k!='的' and k!='有' and not k in tmp_dict.keys():
                isl_dictionary[k] = tmp_dict[word]
                break
    return isl_dictionary