import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../')

import numpy as np
import torch
from torch.autograd import Variable

from pydeformetrica.libs.libkp.python.bindings.torch.kernels import Kernel, KernelProduct
from pydeformetrica.libs.libkp.python.pykp.pytorch.kernel_product import KernelProductGrad_x
from pydeformetrica.src.support.utilities.general_settings import Settings


class CudaExactKernel:

    def __init__(self):
        self.kernel_width = None
        # self.kernel_product = KernelProduct().apply
        self.kernel_product_grad_x = KernelProductGrad_x().apply

    def convolve(self, x, y, p, mode='gaussian(x,y)'):

        assert self.kernel_width != None, "pykp kernel width not initialized"

        kw = Variable(torch.from_numpy(np.array([self.kernel_width])).type(Settings().tensor_scalar_type),
                      requires_grad=False)

        if mode == 'gaussian(x,y)': return KernelProduct(kw, x, y, p, Kernel(mode), 'sum').apply()
        else: return KernelProduct((kw, None), x, y, p, Kernel(mode), 'sum').apply()

    def convolve_gradient(self, px, x, y=None, py=None):

        if y is None: y = x.clone()
        if py is None: py = px.clone()

        kw = Variable(torch.from_numpy(np.array([self.kernel_width])).type(Settings().tensor_scalar_type),
                      requires_grad=False)

        return self.kernel_product_grad_x(kw, px, x, y, py, 'gaussian')


    # def convolve_gradient(self, px, x, y=None, py=None):
    #
    #     # Default values.
    #     if y is None: y = x
    #     if py is None: py = px
    #
    #     # Asserts.
    #     assert (x.size()[0] == px.size()[0])
    #     assert (y.size()[0] == py.size()[0])
    #     assert (px.size()[1] == py.size()[1])
    #     assert (x.size()[1] == y.size()[1])
    #
    #     # temp = torch.autograd.Variable(x.data, requires_grad=True)
    #     x.requires_grad = True
    #     e = torch.ones_like(x).type(Settings().tensor_scalar_type)
    #     out = torch.autograd.grad(self.convolve(x, y, px), x, e, create_graph=True)[0]
    #
    #     # Return.
    #     return out
