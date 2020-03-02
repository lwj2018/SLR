import torch
import numpy
from torchtext.data.metrics import bleu_score
from utils.textUtils import itos,itos_clip

def count_wer(output, tgt):
    """
      shape of output is: T x E or T x N x E
      shape of tgt is:    T or T x N
    """
    if len(output.size()==2):
        output = torch.argmax(output,1)
        output = output.detach().data.cpu().numpy()
        tgt = tgt.detach().data.cpu().numpy()
        return wer(tgt,output)
    elif len(output.size()==3)
        output = torch.argmax(output,2)
        output = output.detach().data.cpu().numpy()
        output = output.transpose(1,0)
        tgt = tgt.detach().data.cpu().numpy()
        tgt = tgt.transpose(1,0)
        total_wer = 0.0
        for o, t in zip(output,tgt):
            total_wer += wer(tgt,output)
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

def count_bleu(output, trg, revese_dict):
    # output shape: seq_len * batch_size * feature
    #           or: MAX_LEN * batch_size (generate by greedy decode) 
    # trg shape: seq_len * batch_size
    # corpus level or sentence level bleu ?
    if len(output.size())==3:
        output = output.permute(1,0,2).max(2)[1]
        candidate_corpus = [itos(idx_list, revese_dict) for idx_list in output]
    elif len(output.size())==2:
        output = output.permute(1,0)
        candidate_corpus = [itos_clip(idx_list, revese_dict) for idx_list in output]
    trg = trg.permute(1,0)
    references_corpus = [[(idx_list, TRG)] for idx_list in trg]
    return bleu_score(candidate_corpus, references_corpus)
