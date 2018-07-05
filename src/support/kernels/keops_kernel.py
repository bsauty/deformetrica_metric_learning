from support.kernels import AbstractKernel
from pykeops.torch import generic_sum
from support.utilities.general_settings import Settings

from pykeops.torch.kernels import Kernel, kernel_product

import logging
logger = logging.getLogger(__name__)


class KeopsKernel(AbstractKernel):
    def __init__(self, kernel_width=None, device='auto'):
        super().__init__(kernel_width, device)
        self.kernel_type = 'keops'

        logger.info('Initializing the Keops kernel for an ambient space of dimension %d.' % Settings().dimension)

        self.gaussian_convolve = generic_sum(
            "Exp(-G*SqDist(X,Y)) * P",
            "O = Vx(" + str(Settings().dimension) + ")",
            "G = Pm(1)",
            "X = Vx(" + str(Settings().dimension) + ")",
            "Y = Vy(" + str(Settings().dimension) + ")",
            "P = Vy(" + str(Settings().dimension) + ")")

        self.point_cloud_convolve = generic_sum(
            "Exp(-G*SqDist(X,Y)) * P",
            "O = Vx(1)",
            "G = Pm(1)",
            "X = Vx(" + str(Settings().dimension) + ")",
            "Y = Vy(" + str(Settings().dimension) + ")",
            "P = Vy(1)")

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

    def convolve(self, x, y, p, mode='gaussian'):

        if mode == 'gaussian':
            return self.gaussian_convolve(self.gamma, x, y, p, backend=self.device)

        elif mode == 'pointcloud':
            return self.point_cloud_convolve(self.gamma, x, y, p, backend=self.device)

        elif mode == 'varifold':
            x, nx = x
            y, ny = y
            return self.varifold_convolve(self.gamma, x, y, nx, ny, p, backend=self.device)

        else:
            raise RuntimeError('Unknown kernel mode.')

    def convolve_gradient(self, px, x, y=None, py=None, mode='gaussian'):

        if y is None:
            y = x
        if py is None:
            py = px

        return -2 * self.gamma * self.gaussian_convolve_gradient_x(self.gamma, x, y, px, py, backend=self.device)
