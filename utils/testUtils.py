import os
import torch
import numpy
import time
from utils.metricUtils import *
from utils.textUtils import itos

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def test_isolated(model, criterion, testloader, device, epoch, log_interval, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # Set eval mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, data in enumerate(testloader):
            # measure data loading time
            data_time.update(time.time() - end)

            # get the inputs and labels
            mat, target = data
            mat = mat.to(device)
            target = target.to(device)

            # forward
            outputs = model(mat)

            # compute the loss
            loss = criterion(outputs, target)

            # compute the metrics
            prec1, prec5 = accuracy(outputs.data, target, topk=(1,5))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # update average value
            losses.update(loss.item())
            top1.update(prec1.item())
            top5.update(prec5.item())
            if i % 50 == 0:
                print("%d/%d %.2f"%(i,len(testloader),top1.avg))

        info = ('[Test] Epoch: [{0}] [len: {1}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Batch Prec@1 {top1.avg:.4f}\t'
                'Batch Prec@5 {top5.avg:.4f}\t'
                .format(
                    epoch, len(testloader), batch_time=batch_time, loss=losses,
                    data_time=data_time,  top1=top1, top5=top5
                    ))
        print(info)
        writer.add_scalar('val acc',
            top1.avg,
            epoch)

    return top1.avg

def test_vae(model, criterion, testloader, device, epoch, log_interval, writer, output_path):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # Set eval mode
    model.eval()
    # create output path
    if not os.path.exists(output_path): os.makedirs(output_path)

    end = time.time()
    with torch.no_grad():
        for i, data in enumerate(testloader):
            # measure data loading time
            data_time.update(time.time() - end)

            # get the inputs and labels
            mat, target = data
            mat = mat.to(device)
            target = target.to(device)

            # forward
            outputs = model.classify(mat)
            recons, input, mu, log_var = model(mat)
            # save recons & input
            if i%100==0:
                recons_save_name = os.path.join(output_path,'recons_%06d.npy'%i)
                recons = recons.detach().data.cpu().numpy()
                numpy.save(recons_save_name,recons)
                input_save_name = os.path.join(output_path,'input_%06d.npy'%i)
                input = input.detach().data.cpu().numpy()
                numpy.save(input_save_name,input)

            # compute the loss
            loss = criterion(outputs, target)

            # compute the metrics
            prec1, prec5 = accuracy(outputs.data, target, topk=(1,5))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # update average value
            losses.update(loss.item())
            top1.update(prec1.item())
            top5.update(prec5.item())

        info = ('[Test] Epoch: [{0}] [len: {1}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Batch Prec@1 {top1.avg:.4f}\t'
                'Batch Prec@5 {top5.avg:.4f}\t'
                .format(
                    epoch, len(testloader), batch_time=batch_time, loss=losses,
                    data_time=data_time,  top1=top1, top5=top5
                    ))
        print(info)
        writer.add_scalar('val acc',
            top1.avg,
            epoch)

    return top1.avg

def test_hcn_lstm(model, criterion, testloader, device, epoch, log_interval, writer, reverse_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_wer = AverageMeter()
    avg_bleu = AverageMeter()
    # Set eval mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, data in enumerate(testloader):
            # measure data loading time
            data_time.update(time.time() - end)

            # get the inputs and labels
            # shape of tgt is N x T
            input, tgt = data['src'].to(device), data['tgt'].to(device)
            src_len_list, tgt_len_list = data['src_len_list'].to(device), data['tgt_len_list'].to(device)

            # forward
            outputs = model(input, src_len_list)
            # print(outputs.argmax(2).permute(1,0))
            # print(tgt)

            # compute the loss
            loss = criterion(outputs,tgt,src_len_list,tgt_len_list)

            # compute the metrics
            wer = count_wer(outputs, tgt)
            bleu = count_bleu(outputs, tgt.permute(1,0), reverse_dict)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # update average value
            N = tgt.size(0)
            losses.update(loss,N)
            avg_wer.update(wer,N)
            avg_bleu.update(bleu,N)

            if i % log_interval == log_interval-1:
                # Warning! when N = 1, have to unsqueeze
                # Qualitative evaluation of translation result
                # outputs = outputs.unsqueeze(1).permute(1,0,2).max(2)[1]
                outputs = outputs.permute(1,0,2).max(2)[1]
                outputs = outputs.data.cpu().numpy()
                outputs = [' '.join(itos_clip(compress(idx_list), reverse_dict)) for idx_list in outputs]
                tgt = tgt.view(-1,tgt.size(-1))
                tgt = tgt.data.cpu().numpy()
                tgt = [' '.join(itos_clip(compress(idx_list), reverse_dict)) for idx_list in tgt]
                writer.add_text('outputs', 
                                str(outputs),
                                epoch * len(testloader) + i)
                writer.add_text('tgt',
                                str(tgt),
                                epoch * len(testloader) + i)

        info = ('[Test] Epoch: [{0}][len: {1}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Batch Wer {wer.avg:.5f}\t'
                'Batch Bleu {bleu.avg:.5f}\t'
                .format(
                    epoch, len(testloader), batch_time=batch_time,
                    data_time=data_time, loss=losses, wer=avg_wer, bleu=avg_bleu
                    ))
        print(info)
        writer.add_scalar('test wer',
            avg_wer.avg,
            epoch * len(testloader) + i)
        writer.add_scalar('test bleu',
            avg_bleu.avg,
            epoch * len(testloader) + i)


    return avg_wer.avg