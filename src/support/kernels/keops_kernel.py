import torch

from support.kernels import AbstractKernel
from pykeops.torch import Kernel, kernel_product
from pykeops.torch import generic_sum
from support.utilities.general_settings import Settings


class KeopsKernel(AbstractKernel):
    def __init__(self, kernel_width=None):
        self.kernel_type = 'keops'
        super().__init__(kernel_width)
        Settings().dimension = 2
        self.convolve = generic_sum("Exp(-G*SqDist(X,Y)) * B",
                                    "A = Vx(" + str(Settings().dimension) + ")",
                                    "G = Pm(1)",
                                    "X = Vx(" + str(Settings().dimension) + ")",
                                    "Y = Vy(" + str(Settings().dimension) + ")",
                                    "B = Vy(" + str(Settings().dimension) + ")")
        self.convolve_gradient_x = generic_sum("(P,X-Y)*Exp(-G*SqDist(X,Y))*B",
                                               "A = Vx(" + str(Settings().dimension) + ")",
                                               "G = Pm(1)",
                                               "X = Vx(" + str(Settings().dimension) + ")",
                                               "Y = Vy(" + str(Settings().dimension) + ")",
                                               "P = Vx(" + str(Settings().dimension) + ")",
                                               "B = Vy(" + str(Settings().dimension) + ")")

    def convolve(self, x, y, p, backend='auto', mode='gaussian(x,y)'):
        kw = torch.tensor([self.kernel_width], dtype=x.dtype)
        return self.convolve(1./kw, x, y, p)

    def convolve_gradient(self, px, x, y=None, py=None, backend='auto', mode='gaussian(x,y)'):

        if y is None:
            y = x
        if py is None:
            py = px

        kw = torch.tensor([self.kernel_width], dtype=x.dtype)
        return -2 * self.convolve_gradient_x(1./kw, x, y, px, py) / kw

    # def convolve(self, x, y, p, backend='auto', mode='gaussian(x,y)'):
    #     assert self.kernel_width != None, "pykeops kernel width not initialized"
    #
    #     kw = torch.tensor([self.kernel_width], dtype=x.dtype, requires_grad=False)
    #
    #     params = {
    #         "id": Kernel(mode),
    #         'gamma': 1. / kw ** 2,
    #         'backend': backend
    #     }
    #
    #     # return kernel_product(x, y, p, params).type(Settings().tensor_scalar_type)
    #     return kernel_product(params, x, y, p).type(Settings().tensor_scalar_type)
    #
    # def convolve_gradient(self, px, x, y=None, py=None, backend='auto', mode='gaussian(x,y)'):
    #
    #     px.requires_grad_(True)
    #     x.requires_grad_(True)
    #
    #     factor = 1.0
    #     if y is None:
    #         y = x
    #         factor = 0.5
    #     if py is None:
    #         py = px
    #
    #     kw = torch.tensor([self.kernel_width], dtype=x.dtype)
    #
    #     params = {
    #         'id': Kernel(mode),
    #         'gamma': 1. / kw ** 2,
    #         'backend': backend
    #     }
    #
    #     px_xKy_py = torch.dot(px.view(-1),
    #                           kernel_product(params, x, y, py).type(Settings().tensor_scalar_type).view(-1))
    #     return factor * torch.autograd.grad(px_xKy_py, [x], create_graph=True)[0]

# class KeopsKernel(AbstractKernel):
#     def __init__(self, kernel_width=None):
#         self.kernel_type = 'keops'
#         self.kernel_width = kernel_width
#
#     def convolve(self, x, y, p, mode='gaussian(x,y)'):
#
#         assert self.kernel_width != None, "pykp kernel width not initialized"
#
#         kw = Variable(torch.from_numpy(np.array([self.kernel_width])).type(Settings().tensor_scalar_type),
#                       requires_grad=False)
#
#         params = {
#             'id': Kernel(mode),
#             'gamma': 1. / kw ** 2 if mode == 'gaussian(x,y)' else (1. / kw ** 2, 1. / kw ** 2),
#             'backend': 'auto'
#         }
#
#         return kernel_product(x, y, p, params).type(Settings().tensor_scalar_type)
#
#     def convolve_gradient(self, px, x, y=None, py=None, mode='gaussian(x,y)'):
#
#         factor = 1.0
#         if y is None:
#             y = x
#             factor = 0.5
#         if py is None:
#             py = px
#
#         kw = Variable(torch.from_numpy(np.array([self.kernel_width])).type(Settings().tensor_scalar_type),
#                       requires_grad=True)
#
#         params = {
#             'id': Kernel(mode),
#             'gamma': 1. / kw ** 2,
#             'backend': 'auto'
#         }
#
#         px_xKy_py = torch.dot(px.view(-1),
#                               kernel_product(x, y, py, params).type(Settings().tensor_scalar_type).view(-1))
#
#         return factor * grad(px_xKy_py, [x], create_graph=True)[0]
#
#     def get_kernel_matrix(self, x, y=None):
#         """
#         returns the kernel matrix, A_{ij} = exp(-|x_i-x_j|^2/sigma^2)
#         """
#         if y is None: y = x
#         assert (x.size()[0] == y.size()[0])
#         sq = self._squared_distances(x, y)
#         return torch.exp(-sq / (self.kernel_width ** 2))
#
#     ####################################################################################################################
#     ### Private methods:
#     ####################################################################################################################
#
#     def _squared_distances(self, x, y):
#         """
#         Returns the matrix of $(x_i - y_j)^2$.
#         Output is of size (1, M, N).
#         """
#         return torch.sum((x.unsqueeze(1) - y.unsqueeze(0)) ** 2, 2)
