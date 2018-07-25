import torch

from support.kernels import AbstractKernel
from pykeops.torch import generic_sum


import logging
logger = logging.getLogger(__name__)


class KeopsKernel(AbstractKernel):
    def __init__(self, kernel_width=None, tensor_scalar_type=None, device='auto', **kwargs):
        super().__init__(kernel_width, tensor_scalar_type, device)
        self.kernel_type = 'keops'
        self.gamma = 1. / torch.tensor([self.kernel_width ** 2]).type(self.tensor_scalar_type)

        self.gaussian_convolve = []
        self.point_cloud_convolve = []
        self.varifold_convolve = []
        self.gaussian_convolve_gradient_x = []

        for dimension in [2, 3]:
            self.gaussian_convolve.append(generic_sum(
                "Exp(-G*SqDist(X,Y)) * P",
                "O = Vx(" + str(dimension) + ")",
                "G = Pm(1)",
                "X = Vx(" + str(dimension) + ")",
                "Y = Vy(" + str(dimension) + ")",
                "P = Vy(" + str(dimension) + ")"))

            self.point_cloud_convolve.append(generic_sum(
                "Exp(-G*SqDist(X,Y)) * P",
                "O = Vx(1)",
                "G = Pm(1)",
                "X = Vx(" + str(dimension) + ")",
                "Y = Vy(" + str(dimension) + ")",
                "P = Vy(1)"))

            self.varifold_convolve.append(generic_sum(
                "Exp(-(WeightedSqDist(G, X, Y))) * Pow((Nx, Ny), 2) * P",
                "O = Vx(1)",
                "G = Pm(1)",
                "X = Vx(" + str(dimension) + ")",
                "Y = Vy(" + str(dimension) + ")",
                "Nx = Vx(" + str(dimension) + ")",
                "Ny = Vy(" + str(dimension) + ")",
                "P = Vy(1)"))

            self.gaussian_convolve_gradient_x.append(generic_sum(
                "(Px, Py) * Exp(-G*SqDist(X,Y)) * (X-Y)",
                "O = Vx(" + str(dimension) + ")",
                "G = Pm(1)",
                "X = Vx(" + str(dimension) + ")",
                "Y = Vy(" + str(dimension) + ")",
                "Px = Vx(" + str(dimension) + ")",
                "Py = Vy(" + str(dimension) + ")"))

    def convolve(self, x, y, p, mode='gaussian'):
        dimension = x.size(1)
        self._check_tensor_device(self.gamma.device, self.gamma, x, y, p)
        if mode == 'gaussian':
            return self.gaussian_convolve[dimension - 2](self.gamma, x, y, p, backend=self.device)

        elif mode == 'pointcloud':
            return self.point_cloud_convolve[dimension - 2](self.gamma, x, y, p, backend=self.device)

        elif mode == 'varifold':
            x, nx = x
            y, ny = y
            return self.varifold_convolve[dimension - 2](self.gamma, x, y, nx, ny, p, backend=self.device)

        else:
            raise RuntimeError('Unknown kernel mode.')

    def convolve_gradient(self, px, x, y=None, py=None, mode='gaussian'):
        if y is None:
            y = x
        if py is None:
            py = px

        d = x.size(1)
        self._check_tensor_device(self.gamma.device, self.gamma, px, x, y, py)
        return -2 * self.gamma * self.gaussian_convolve_gradient_x[d - 2](self.gamma, x, y, px, py, backend=self.device)
