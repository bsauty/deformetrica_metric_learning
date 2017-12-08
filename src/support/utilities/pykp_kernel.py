import torch
import sys
import os
from torch.autograd import Variable
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')
from pydeformetrica.src.support.utilities.singleton_pattern import Singleton
from pykp.pytorch.kernel_product import KernelProduct


#deux choix : pytorch pur ou pytorch version benji (cuda only so far TODO : implementation cpu type deformetrica)



class PYKPKernel:
    def __init__(self):
        self.KernelWidth = None

    def Convolve(self,x,p,y):
        assert self.KernelWidth != None, "pykp kernel width not initialized"
        return KernelProduct(s, x, y, p, mode)

    def ConvolveGradient(self,x,p,y):
        H = torch.dot(p.view(-1), self.Convolve(x,p,y).view(-1))
        return torch.autograd.grad(H, p, create_graph=True)[0]
