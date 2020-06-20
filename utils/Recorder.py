import time
class Recorder:
    def __init__(self,averagers,names,writer,
        batch_timer=None,data_timer=None):
        self.averagers = averagers
        self.names = names
        self.writer = writer
        self.batch_timer = batch_timer
        self.data_timer = data_timer

    def update(self,vals,count=1):
        for val, averager in zip(vals,self.averagers):
            averager.update(val,count)

    def reset(self):
        for averager in self.averagers:
            averager.reset()
        if self.batch_timer is not None:
            self.batch_timer.reset()
        if self.data_timer is not None:
            self.data_timer.reset()

    def log(self,epoch,iter,l,mode='Train'):
        # log to terminal
        prefix = '[' + mode + '] '
        info = prefix + '[{0}][{1}/{2}]\t'.format(epoch,iter,l)
        if self.batch_timer is not None:
            info = info + 'Time {timer.avg:.3f}s\t'.format(timer=self.batch_timer)
        if self.data_timer is not None:
            info = info + 'Data {timer.avg:.3f}s\t'.format(timer=self.data_timer)
        for name, averager in zip(self.names,self.averagers):
            info = info + name +  ' {x.avg:.4f}\t'.format(x=averager)
        print(info)
        # log to tensorboard
        for name, averager in zip(self.names,self.averagers):
            self.writer.add_scalar(name,
                    averager.avg,
                    epoch * l + iter)

    def tik(self):
        self.start = time.time()

    def tok(self):
        self.end = time.time()
        delta = self.end - self.start
        self.batch_timer.update(delta)

    def data_tik(self):
        self.data_start = time.time()

    def data_tok(self):
        self.data_end = time.time()
        delta = self.data_end - self.data_start
        self.data_timer.update(delta)

    def get_avg(self, query_name):
        for name, averager in zip(self.names,self.averagers):
            if name == query_name:
                return averager.avg
        return -1