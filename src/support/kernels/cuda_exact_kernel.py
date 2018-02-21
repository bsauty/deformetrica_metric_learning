import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../')

import numpy as np
import torch
from torch.autograd import Variable

from pydeformetrica.libs.libkp.python.bindings.torch.kernels import Kernel, KernelProduct, kernel_product
from pydeformetrica.libs.libkp.python.pykp.pytorch.kernel_product import KernelProductGrad_x
from pydeformetrica.src.support.utilities.general_settings import Settings


class CudaExactKernel:
    def __init__(self):
        self.kernel_width = None
        # self.kernel_product = KernelProduct().apply
        self.kernel_product_grad_x = KernelProductGrad_x().apply

    # def convolve(self, x, y, p, mode='gaussian(x,y)'):
    #
    #     assert self.kernel_width != None, "pykp kernel width not initialized"
    #
    #     kw = Variable(torch.from_numpy(np.array([self.kernel_width])).type(Settings().tensor_scalar_type),
    #                   requires_grad=False)
    #
    #     return self.kernel_product(kw, x, y, p, 'gaussian')
    #
    #     # if mode == 'gaussian(x,y)': return KernelProduct(kw, x, y, p, Kernel(mode), 'sum').apply()
    #     # else: return KernelProduct((kw, None), x, y, p, Kernel(mode), 'sum').apply()

    def convolve(self, x, y, p, mode='gaussian(x,y)'):

        assert self.kernel_width != None, "pykp kernel width not initialized"

        kw = Variable(torch.from_numpy(np.array([self.kernel_width])).type(Settings().tensor_scalar_type),
                      requires_grad=False)

        params = {
            'id': Kernel(mode),
            'gamma': 1. / kw ** 2 if mode == 'gaussian(x,y)' else (1. / kw ** 2, 1. / kw ** 2),
            'backend': 'auto'
        }

        return kernel_product(x, y, p, params)

    def convolve_gradient(self, px, x, y=None, py=None):

        if y is None: y = x.clone()
        if py is None: py = px.clone()

        kw = Variable(torch.from_numpy(np.array([self.kernel_width])).type(Settings().tensor_scalar_type),
                      requires_grad=False)

        return self.kernel_product_grad_x(kw, px, x, y, py, 'gaussian')
