import torch
import torch.nn.functional as F
import time
from utils.metricUtils import count_wer, count_bleu
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

def train(model, criterion, optimizer, trainloader, device, epoch, logger, log_interval, writer, reverse_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_wer = AverageMeter()
    avg_bleu = AverageMeter()
    # Set trainning mode
    model.train()

    end = time.time()
    for i, data in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        # get the inputs and labels
        # shape of tgt is N x T
        input, tgt = data['input'].to(device), data['tgt'].to(device)

        optimizer.zero_grad()
        # forward
        outputs = model(input, torch.zeros(tgt[:,:-1].size(),dtype=torch.long))

        # compute the loss
        loss = criterion(outputs.view(-1, outputs.shape[-1]), tgt[:,1:].view(-1))

        # backward & optimize
        loss.backward()
        optimizer.step()

        # compute the metrics
        wer = count_wer(outputs, tgt[:,1:])
        bleu = count_bleu(outputs, tgt[:,1:].permute(1,0), reverse_dict)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # update average value
        losses.update(loss.item())
        avg_wer.update(wer)
        avg_bleu.update(bleu)

        if i % log_interval == log_interval-1:
            info = ('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'
                    'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Wer {wer.val:.4f} ({wer.avg:.4f})\t'
                    .format(
                        epoch, i, len(trainloader), batch_time=batch_time,
                        data_time=data_time, loss=losses,  wer=avg_wer,
                        lr=optimizer.param_groups[-1]['lr']))
            print(info)
            writer.add_scalar('train loss',
                    losses.avg,
                    epoch * len(trainloader) + i)
            writer.add_scalar('train wer',
                    avg_wer.avg,
                    epoch * len(trainloader) + i)
            writer.add_scalar('train bleu',
                    avg_bleu.avg,
                    epoch * len(trainloader) + i)
            logger.info("epoch {:3d} | iteration {:5d} | Loss {:.6f}".format(epoch+1, i+1, losses.avg))
            # Reset average meters 
            losses.reset()
            avg_wer.reset()
            avg_bleu.reset()

            # # Qualitative evaluation of translation result
            # outputs = outputs.permute(1,0,2).max(2)[1]
            # outputs = outputs.data.cpu().numpy()
            # outputs = [' '.join(itos(idx_list, reverse_dict)) for idx_list in outputs]
            # tgt = tgt.view(-1,tgt.size(-1))
            # tgt = tgt.data.cpu().numpy()
            # tgt = [' '.join(itos(idx_list, reverse_dict)) for idx_list in tgt]
            # writer.add_text('train_outputs', 
            #                 str(outputs),
            #                 epoch * len(trainloader) + i)
            # writer.add_text('train_tgt',
            #                 str(tgt),
            #                 epoch * len(trainloader) + i)

