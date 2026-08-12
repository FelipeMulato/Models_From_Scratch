import torch


class SGD:
    def __init__(self,parms,lr):
        self.parms = parms
        self.lr = lr
    def step(self):
        for parm in self.parms:
            parm -= parm.grad()*self.lr
    def zero_grad(self):
        for parm in self.parms:
            if parm.grad is not None:
                parm.zero_grad()