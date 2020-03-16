import torch

def skeleton_collate(batch):
    sources = []
    targets = []
    src_len_list = []
    tgt_len_list = []
    max_src_len = 0
    max_tgt_len = 0
    # shape of source is: s x clip_length x J x D
    # shape of target is: t
    for sample in batch:
        source = torch.Tensor(sample['input'])
        target = torch.LongTensor(sample['tgt'])
        # remember max src len
        src_len = source.size(0)
        if src_len > max_src_len:
            max_src_len = src_len
        # remember max tg len
        tgt_len = target.size(0)
        if tgt_len > max_tgt_len:
            max_tgt_len = tgt_len
        # append
        src_len_list.append(src_len)
        tgt_len_list.append(tgt_len)
        sources.append(source)
        targets.append(target)
    # Fill zeros
    full_sources = []
    full_targets = []
    # sort by src len
    X = zip(sources, targets, src_len_list, tgt_len_list)
    X = sorted(X, key=lambda t:t[2], reverse=True)
    # rearrange len list
    src_len_list = []
    tgt_len_list = []
    for source, target, src_len, tgt_len in X:
        # pad the source
        src_pad = torch.zeros( (max_src_len-src_len,) + source.size()[-3:] )
        full_src = torch.cat([source, src_pad], 0)
        full_sources.append(full_src)
        # pad the target
        tgt_pad = torch.zeros(max_tgt_len-tgt_len,dtype=torch.long)
        full_tgt = torch.cat([target,tgt_pad], 0)
        full_targets.append(full_tgt)
        # len list append new item
        src_len_list.append(src_len)
        tgt_len_list.append(tgt_len)
    # After data processing,
    # shape of src is N x S x clip_lenth x J x D, where S is max length of this batch
    # shape of tgt is N x T, where T is max length of this batch
    # shape of src_len_list is N
    # shape of tgt_len_list is N
    src = torch.stack(full_sources, 0)
    tgt = torch.stack(full_targets, 0)
    src_len_list = torch.LongTensor(src_len_list)
    tgt_len_list = torch.LongTensor(tgt_len_list)
    return {'src':src, 'tgt':tgt, 'src_len_list':src_len_list, 'tgt_len_list':tgt_len_list}


def rgb_collate(batch):
    sources = []
    targets = []
    src_len_list = []
    tgt_len_list = []
    max_src_len = 0
    max_tgt_len = 0
    # shape of source is s x C x 16 x H x W
    # shape of target is t
    for sample in batch:
        source = torch.Tensor(sample['input'])
        target = torch.LongTensor(sample['tgt'])
        # remember max src len
        src_len = source.size(0)
        if src_len > max_src_len:
            max_src_len = src_len
        # remember max tg len
        tgt_len = target.size(0)
        if tgt_len > max_tgt_len:
            max_tgt_len = tgt_len
        # append
        src_len_list.append(src_len)
        tgt_len_list.append(tgt_len)
        sources.append(source)
        targets.append(target)
    # fill zeros
    full_sources = []
    full_targets = []
    # sort by src len
    X = zip(sources, targets, src_len_list, tgt_len_list)
    X = sorted(X, key=lambda t:t[2], reverse=True)
    # rearrange len list
    src_len_list = []
    tgt_len_list = []
    for source, target, src_len, tgt_len in X:
        # pad the source
        src_pad = torch.zeros( (max_src_len-src_len,) + source.size()[-4:] )
        full_src = torch.cat([source, src_pad], 0)
        full_sources.append(full_src)
        # pad the target
        tgt_pad = torch.zeros(max_tgt_len-tgt_len,dtype=torch.long)
        full_tgt = torch.cat([target,tgt_pad], 0)
        full_targets.append(full_tgt)
        # len list append new item
        src_len_list.append(src_len)
        tgt_len_list.append(tgt_len)
    # After data processing,
    # shape of src is N x S x C x 16 x H x W, where S is max length of this batch
    # shape of tgt is N x T, where T is max length of this batch
    # shape of src_len_list is N
    # shape of tgt_len_list is N
    src = torch.stack(full_sources, 0)
    tgt = torch.stack(full_targets, 0)
    src_len_list = torch.LongTensor(src_len_list)
    tgt_len_list = torch.LongTensor(tgt_len_list)
    return {'src':src, 'tgt':tgt, 'src_len_list':src_len_list, 'tgt_len_list':tgt_len_list}
