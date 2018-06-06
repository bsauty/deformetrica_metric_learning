from support.kernels import AbstractKernel
from pykeops.torch import generic_sum
from support.utilities.general_settings import Settings
import torch

class KeopsKernel(AbstractKernel):
    def __init__(self, kernel_width=None):
        self.kernel_type = 'keops'
        super().__init__(kernel_width)

        self.gaussian_convolve = generic_sum(
            "Exp(-G*SqDist(X,Y)) * P",
            "O = Vx(" + str(Settings().dimension) + ")",
            "G = Pm(1)",
            "X = Vx(" + str(Settings().dimension) + ")",
            "Y = Vy(" + str(Settings().dimension) + ")",
            "P = Vy(" + str(Settings().dimension) + ")")

        self.varifold_convolve = generic_sum(
            "Exp(-(WeightedSqDist(G, X, Y))) * Pow((Nx, Ny), 2) * P",
            "O = Vx(1)",
            "G = Pm(1)",
            "X = Vx(" + str(Settings().dimension) + ")",
            "Y = Vy(" + str(Settings().dimension) + ")",
            "Nx = Vx(" + str(Settings().dimension) + ")",
            "Ny = Vy(" + str(Settings().dimension) + ")",
            "P = Vy(1)")

        self.gaussian_convolve_gradient_x = generic_sum(
            "(Px, Py) * Exp(-G*SqDist(X,Y)) * (X-Y)",
            "O = Vx(" + str(Settings().dimension) + ")",
            "G = Pm(1)",
            "X = Vx(" + str(Settings().dimension) + ")",
            "Y = Vy(" + str(Settings().dimension) + ")",
            "Px = Vx(" + str(Settings().dimension) + ")",
            "Py = Vy(" + str(Settings().dimension) + ")")

    def convolve(self, x, y, p, backend='auto', mode='gaussian'):
        if mode == 'gaussian':
            return self.gaussian_convolve(self.gamma, x, y, p, backend=backend)

        elif mode == 'varifold':
            cx, nx = x
            cy, ny = y
            return self.varifold_convolve(self.gamma, cx, cy, nx, ny, p, backend=backend)

        else:
            raise RuntimeError('Unknown kernel mode.')

    def convolve_gradient(self, px, x, y=None, py=None, backend='auto', mode='gaussian'):

        if y is None:
            y = x
        if py is None:
            py = px

        return -2 * self.gamma * self.gaussian_convolve_gradient_x(self.gamma, x, y, px, py, backend=backend)




# # OTHER VERSION
# from support.kernels import AbstractKernel
# from pykeops.torch import generic_sum, Kernel, kernel_product
# from support.utilities.general_settings import Settings
# import torch
#
#
# class KeopsKernel(AbstractKernel):
#     def __init__(self, kernel_width=None):
#         self.kernel_type = 'keops'
#         super().__init__(kernel_width)
#
#         self.gaussian_convolve = generic_sum(
#             "Exp(-G*SqDist(X,Y)) * P",
#             "O = Vx(" + str(Settings().dimension) + ")",
#             "G = Pm(1)",
#             "X = Vx(" + str(Settings().dimension) + ")",
#             "Y = Vy(" + str(Settings().dimension) + ")",
#             "P = Vy(" + str(Settings().dimension) + ")")
#
#         self.varifold_convolve = generic_sum(
#             "Exp(-(WeightedSqDist(G, X, Y))) * Square(Nx, Ny) * P",
#             "O = Vx(1)",
#             "G = Pm(1)",
#             "X = Vx(" + str(Settings().dimension) + ")",
#             "Y = Vy(" + str(Settings().dimension) + ")",
#             "Nx = Vx(" + str(Settings().dimension) + ")",
#             "Ny = Vy(" + str(Settings().dimension) + ")",
#             "P = Vy(1)")
#
#         self.gaussian_convolve_gradient_x = generic_sum(
#             "(Px, Py) * Exp(-G*SqDist(X,Y)) * (X-Y)",
#             "O = Vx(" + str(Settings().dimension) + ")",
#             "G = Pm(1)",
#             "X = Vx(" + str(Settings().dimension) + ")",
#             "Y = Vy(" + str(Settings().dimension) + ")",
#             "Px = Vx(" + str(Settings().dimension) + ")",
#             "Py = Vy(" + str(Settings().dimension) + ")")
#
#     def _convolve_with_grad(self, x, y, p, backend='auto', mode='gaussian'):
#
#         kw = torch.tensor([self.kernel_width], dtype=torch.float, requires_grad=False)
#
#         params = {
#             "id": Kernel(mode + '(x,y)'),
#             'gamma': 1. / kw ** 2 if mode == 'gaussian(x,y)' else (1. / kw ** 2, 1. / kw ** 2),
#             'backend': backend
#         }
#
#         if mode == 'varifold':
#             params["id"] = "gaussian(x,y) * (linear(u,v)**2) "
#
#         return kernel_product(params, x, y, p)
#
#     def _convolve_without_grad(self, x, y, p, backend='auto', mode='gaussian'):
#         if mode == 'gaussian':
#             return self.gaussian_convolve(self.gamma, x, y, p, backend=backend)
#
#         elif mode == 'varifold':
#             x, nx = x
#             y, ny = y
#             return self.varifold_convolve(self.gamma, x, y, nx, ny, p, backend=backend)
#
#         else:
#             raise RuntimeError('Unknown kernel mode.')
#
#     def convolve(self, x, y, p, backend='auto', mode='gaussian'):
#         assert self.kernel_width != None, "pykeops kernel width not initialized"
#
#         if x.requires_grad or y.requires_grad or p.requires_grad:  # TODO: remove this 'if else'
#             return self._convolve_with_grad(x, y, p, backend, mode)
#
#         else:
#             return self._convolve_without_grad(x, y, p, backend, mode)
#
#     def _convolve_gradient_without_grad(self, px, x, y=None, py=None, backend='auto'):
#         return -2 * self.gamma * self.gaussian_convolve_gradient_x(self.gamma, x, y, px, py, backend=backend)
#
#     def _convolve_gradient_with_grad(self, px, x, y=None, py=None, backend='auto', mode='gaussian'):
#         factor = 1.0
#         if y is None:
#             y = x
#             factor = 0.5
#         if py is None:
#             py = px
#
#         kw = torch.tensor([self.kernel_width], dtype=torch.float, requires_grad=True)
#
#         params = {
#             'id': Kernel(mode + '(x,y)'),
#             'gamma': 1. / kw ** 2,
#             'backend': backend
#         }
#
#         if mode == 'varifold':
#             params["id"] = "gaussian(x,y) * (linear(u,v)**2)"
#
#         # Hamiltonian
#         px_xKy_py = torch.dot(px.view(-1), kernel_product(params, x, y, py).view(-1))
#         return factor * torch.autograd.grad(px_xKy_py, [x], create_graph=True)[0]
#
#     def convolve_gradient(self, px, x, y=None, py=None, backend='auto', mode='gaussian'):
#         if y is None:
#             y = x
#         if py is None:
#             py = px
#
#         if x.requires_grad or y.requires_grad or px.requires_grad:
#             return self._convolve_gradient_with_grad(px, x, y, py=py, backend=backend, mode=mode)
#
#         else:
#             return self._convolve_gradient_without_grad(px, x, y, py=py, backend=backend)
#
#
#
#
#
