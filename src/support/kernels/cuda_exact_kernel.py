import torch
import sys
import os
import numpy as np
from torch.autograd import Variable

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')
from libs.libkp.python.pykp.pytorch.kernel_product import KernelProduct


# deux choix : pytorch pur ou pytorch version benji (cuda only so far TODO : implementation cpu type deformetrica)


class CudaExactKernel:

    def __init__(self):
        self.kernel_width = None
        self.mode = "gaussian"
        self.kernel_product = KernelProduct().apply

    def convolve(self, x, y, p):

        # Asserts.
        assert self.kernel_width != None, "pykp kernel width not initialized"
        
        # Return.
        temp = torch.autograd.Variable(torch.from_numpy(np.array([self.kernel_width])), requires_grad=False )
        out = self.kernel_product(temp, x, y, p, self.mode)
        return out


    def convolve_gradient(self, px, x, y=None, py=None):

        # Default values.
        if y is None: y = x
        if py is None: py = px

        # Asserts.
        assert (x.size()[0] == px.size()[0])
        assert (y.size()[0] == py.size()[0])
        assert (px.size()[1] == py.size()[1])
        assert (x.size()[1] == y.size()[1])

        temp = torch.autograd.Variable(x.data, requires_grad=True )
        e = torch.ones_like(x)
        out =  torch.autograd.grad(self.convolve(temp, y, px),temp,e,create_graph = True)[0]
        print(type(out.data))

        import time

        from pydeformetrica.src.support.utilities.general_settings import Settings
        print(Settings().tensor_scalar_type)



        # Return.
        return out
