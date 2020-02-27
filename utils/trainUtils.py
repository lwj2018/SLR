import torch
import time
from utils.metricUtils import count_wer


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

def train(model, criterion, optimizer, trainloader, device, epoch, logger, log_interval, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_wer = AverageMeter()
    # Set trainning mode
    model.train()

    end = time.time()
    for i, data in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        # get the inputs and labels
        images, tgt = data['images'].to(device), data['sentence'].to(device)

        optimizer.zero_grad()
        # forward
        outputs = model(images, tgt)

        # compute the loss
        loss = criterion(outputs.view(-1, outputs.shape[-1]), tgt.view(-1))

        # compute the WER metrics
        wer = count_wer(outputs.view(-1, outputs.shape[-1]), tgt.view(-1))

        # backward & optimize
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # update average value
        losses.update(loss)
        avg_wer.update(wer)

        if i % log_interval == 0:
            output = ('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'
                    'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Wer {wer.val:.4f} ({wer.avg:.4f})\t'
                    .format(
                        epoch, i, len(trainloader), batch_time=batch_time,
                        data_time=data_time, loss=losses,  wer=avg_wer,
                        lr=optimizer.param_groups[-1]['lr']))
            print(output)

            logger.info("epoch {:3d} | iteration {:5d} | Loss {:.6f}".format(epoch+1, i+1, losses.avg))
    
