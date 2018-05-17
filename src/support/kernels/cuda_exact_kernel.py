import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../')

import numpy as np
import torch
from torch.autograd import Variable, grad

from pydeformetrica.libs.libkp.python.bindings.torch.kernels import Kernel, kernel_product
# from pydeformetrica.libs.libkp.python.pykp.pytorch.kernel_product import KernelProductGrad_x
from pydeformetrica.src.support.utilities.general_settings import Settings


class CudaExactKernel:
    def __init__(self):
        self.kernel_type = 'cudaexact'
        self.kernel_width = None
        # self.kernel_product_grad_x = KernelProductGrad_x().apply

    # def convolve(self, x, y, p, mode='gaussian(x,y)'):
    #
    #     assert self.kernel_width != None, "pykp kernel width not initialized"
    #
    #     kw = Variable(torch.from_numpy(np.array([self.kernel_width])).choisitype(Settings().tensor_scalar_type),
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

        return kernel_product(x, y, p, params).type(Settings().tensor_scalar_type)

    # def convolve_gradient(self, px, x, y=None, py=None):
    #
    #     if y is None: y = x
    #     if py is None: py = px
    #
    #     kw = Variable(torch.from_numpy(np.array([self.kernel_width])).type(Settings().tensor_scalar_type),
    #                   requires_grad=False)
    #
    #     return self.kernel_product_grad_x(kw, px, x, y, py, 'gaussian').type(Settings().tensor_scalar_type)

    def convolve_gradient(self, px, x, y=None, py=None, mode='gaussian(x,y)'):

        if y is None: y = x
        if py is None: py = px

        kw = Variable(torch.from_numpy(np.array([self.kernel_width])).type(Settings().tensor_scalar_type),
                      requires_grad=True)

        params = {
            'id': Kernel(mode),
            'gamma': 1. / kw ** 2,
            'backend': 'auto'
        }

        px_xKy_py = torch.dot(px.view(-1),
                              kernel_product(x, y, py, params).type(Settings().tensor_scalar_type).view(-1))

        return grad(px_xKy_py, [x], create_graph=True)[0]

    def get_kernel_matrix(self, x, y=None):
        """
        returns the kernel matrix, A_{ij} = exp(-|x_i-x_j|^2/sigma^2)
        """
        if y is None: y = x
        assert (x.size()[0] == y.size()[0])
        sq = self._squared_distances(x, y)
        return torch.exp(-sq / (self.kernel_width ** 2))

    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    def _squared_distances(self, x, y):
        """
        Returns the matrix of $(x_i - y_j)^2$.
        Output is of size (1, M, N).
        """
        return torch.sum((x.unsqueeze(1) - y.unsqueeze(0)) ** 2, 2)
