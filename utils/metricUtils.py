import torch
import numpy
from torchtext.data.metrics import bleu_score
from utils.textUtils import itos,itos_clip,compress

def early_stop(sequence):
    if type(sequence[0])==int:
        stop_token = 2
    else: stop_token = '<eos>'
    new_sequence = []
    for token in sequence:
        new_sequence.append(token)
        if token == stop_token:
            break
    return new_sequence

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    # return the k largest elements of the given input Tensor
    # along the given dimension. dim = 1
    # pred is the indices
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def count_bleu(output, trg, reverse_dict):
    # output shape: T * N * vocab_size
    #           or: T * vocab_size (generate by greedy decode) 
    # trg shape: T * N
    # corpus level or sentence level bleu ?
    if len(output.size())==3:
        output = output.permute(1,0,2).max(2)[1]
        output = output.data.cpu().numpy()
        # Early stop & remove padding
        candidate_corpus = [itos_clip(compress(idx_list), reverse_dict) for idx_list in output]
    elif len(output.size())==2:
        output = output.argmax(1).unsqueeze(0)
        output = output.data.cpu().numpy()
        candidate_corpus = [itos_clip(compress(idx_list), reverse_dict) for idx_list in output]
    trg = trg.permute(1,0)
    trg = trg.data.cpu().numpy()
    references_corpus = [[itos(idx_list, reverse_dict)] for idx_list in trg]
    return bleu_score(candidate_corpus, references_corpus)

def count_wer(output, tgt):
    """
      shape of output is: T x vocab_size or T x N x vocab_size
      shape of tgt is:    T or N x T
    """
    if len(output.size())==2:
        output = torch.argmax(output,1)
        output = output.detach().data.cpu().numpy()
        tgt = tgt.detach().data.cpu().numpy().reshape(-1)
        return wer(tgt,output)
    elif len(output.size())==3:
        output = torch.argmax(output,2)
        output = output.detach().data.cpu().numpy()
        output = output.transpose(1,0)
        tgt = tgt.detach().data.cpu().numpy()
        total_wer = 0.0
        for o, t in zip(output,tgt):
            # early stop
            o = numpy.array(early_stop(o))
            # ignore padding
            # o = o[o!=0]
            t = t[t!=0]
            # compress
            o = compress(o)
            t = compress(t)
            total_wer += wer(t,o)
        avg_wer = total_wer/output.shape[0]
        return avg_wer

def wer(r, h):
    """
    Calculation of WER with Levenshtein distance.

    Works only for iterables up to 254 elements (uint8).
    O(nm) time ans space complexity.

    Parameters
    ----------
    r : reference list
    h : hypothese list

    Returns
    -------
    int

    Examples
    --------
    >>> wer("who is there".split(), "is there".split())
    1
    >>> wer("who is there".split(), "".split())
    3
    >>> wer("".split(), "who is there".split())
    3
    """
    # initialisation
    import numpy
    # if len(r) == 0:
    #     print('Warning! len of reference is 0')
    d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion    = d[i][j-1] + 1
                deletion     = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)]/len(r)


