import torch
from pykeops.torch.generic_red import generic_sum

from support.kernels import AbstractKernel


import logging
logger = logging.getLogger(__name__)


class KeopsKernel(AbstractKernel):
    def __init__(self, kernel_width=None, device='auto', dimension=2, tensor_scalar_type=torch.FloatTensor, **kwargs):
        super().__init__(kernel_width, device)
        self.kernel_type = 'keops'
        self.dimension = dimension
        self.tensor_scalar_type = tensor_scalar_type
        self.gamma = 1. / torch.tensor([self.kernel_width ** 2]).type(self.tensor_scalar_type)

        logger.info('Initializing the Keops kernel for an ambient space of dimension %d.' % self.dimension)

        self.gaussian_convolve = generic_sum(
            "Exp(-G*SqDist(X,Y)) * P",
            ["G = Pm(1)", "X = Vx(" + str(self.dimension) + ")", "Y = Vy(" + str(self.dimension) + ")", "P = Vy(" + str(self.dimension) + ")"],
            backend=self.device)

        self.point_cloud_convolve = generic_sum(
            "Exp(-G*SqDist(X,Y)) * P",
            ["G = Pm(1)", "X = Vx(" + str(self.dimension) + ")", "Y = Vy(" + str(self.dimension) + ")", "P = Vy(1)"],
            backend=self.device)

        self.varifold_convolve = generic_sum(
            "Exp(-(WeightedSqDist(G, X, Y))) * Square((Nx|Ny)) * P",
            ["G = Pm(1)", "X = Vx(" + str(self.dimension) + ")", "Y = Vy(" + str(self.dimension) + ")", "Nx = Vx(" + str(self.dimension) + ")", "Ny = Vy(" + str(self.dimension) + ")", "P = Vy(1)"],
            backend=self.device)

        self.gaussian_convolve_gradient_x = generic_sum(
            "(Px| Py) * Exp(-G*SqDist(X,Y)) * (X-Y)",
            ["G = Pm(1)", "X = Vx(" + str(self.dimension) + ")", "Y = Vy(" + str(self.dimension) + ")", "Px = Vx(" + str(self.dimension) + ")", "Py = Vy(" + str(self.dimension) + ")"],
            backend=self.device)

    def convolve(self, x, y, p, mode='gaussian'):
        self._check_tensor_device(self.gamma.device, self.gamma, x, y, p)
        if mode == 'gaussian':
            return self.gaussian_convolve(self.gamma, x, y, p)

        elif mode == 'pointcloud':
            return self.point_cloud_convolve(self.gamma, x, y, p)

        elif mode == 'varifold':
            x, nx = x
            y, ny = y
            return self.varifold_convolve(self.gamma, x, y, nx, ny, p)

        else:
            raise RuntimeError('Unknown kernel mode.')

    def convolve_gradient(self, px, x, y=None, py=None, mode='gaussian'):
        if y is None:
            y = x
        if py is None:
            py = px

        self._check_tensor_device(self.gamma.device, self.gamma, px, x, y, py)
        return -2 * self.gamma * self.gaussian_convolve_gradient_x(self.gamma, x, y, px, py)
