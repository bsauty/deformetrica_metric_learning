import torch
import sys
import os
from torch.autograd import Variable
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')
from pydeformetrica.src.support.utilities.singleton_pattern import Singleton
from pydeformetrica.src.support.utilities.kernel_types import KernelType


class TorchKernel:
    def __init__(self):
        self.KernelWidth = None

    def _squared_distances(self, x, y):
    	"""
        Returns the matrix of $\|x_i-y_j\|^2$.
        """
    	x_col = x.unsqueeze(1) # (N,D) -> (N,1,D)
    	y_lin = y.unsqueeze(0) # (M,D) -> (1,M,D)
        #output is of size (1,M,N)
        # print(type(x_col), type(y_lin))
    	return torch.sum((x_col - y_lin)**2,2)

    def Convolve(self,x,p,y):
        assert self.KernelWidth != None, "torch kernel width not initialized"
        sq = self._squared_distances(x,y)
        return torch.mm(torch.exp(-sq/(self.KernelWidth**2)),p)
        #TODO this is the pure pytorch implementation of the convolution
        #See libkp for a cuda/pytorch implementation

    def ConvolveGradient(self,x,p,y):
        #Hamiltonian
        H = torch.dot(p.view(-1), self.
        Convolve(x,p,y).view(-1))
        return torch.autograd.grad(H, p, create_graph=True)[0]
