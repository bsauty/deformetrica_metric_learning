import torch
import sys
import os
from torch.autograd import Variable

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')
from pydeformetrica.src.support.utilities.singleton_pattern import Singleton
from libs.pykp.pytorch.kernel_product import KernelProduct, KernelProductGrad_x


# deux choix : pytorch pur ou pytorch version benji (cuda only so far TODO : implementation cpu type deformetrica)



class CudaTorchKernel:

    def __init__(self):
        self.kernel_width = None

    def Convolve(self, x, y, p):

        # Asserts.
        assert self.kernel_width != None, "pykp kernel width not initialized"

        # Return.
        return KernelProduct(s, x, y, p, mode)


    def ConvolveGradient(self, px, x, y=None, py=None):

        # Default values.
        if y is None: y = x
        if py is None: py = px

        # Asserts.
        assert (x.size()[0] == px.size()[0])
        assert (y.size()[0] == py.size()[0])
        assert (px.size()[1] == py.size()[1])
        assert (x.size()[1] == y.size()[1])

        # Return.
        return KernelProductGrad_x(s, a, x, y, p, mode)