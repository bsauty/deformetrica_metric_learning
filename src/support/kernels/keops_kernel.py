import torch

from support import utilities
from support.kernels import AbstractKernel
from pykeops.torch import generic_sum
# from pykeops.torch.generic.generic_red import Genred as generic_sum
from core import default


import logging


logger = logging.getLogger(__name__)


class KeopsKernel(AbstractKernel):
    def __init__(self, kernel_width=None, device=default.deformation_kernel_device, cuda_type=default.dtype, **kwargs):

        if device.lower() == 'cuda':
            device = 'GPU'

        super().__init__('keops', kernel_width, device)

        self.gamma = 1. / default.tensor_scalar_type([self.kernel_width ** 2])

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
                "P = Vy(" + str(dimension) + ")", cuda_type=cuda_type))

            self.point_cloud_convolve.append(generic_sum(
                "Exp(-G*SqDist(X,Y)) * P",
                "O = Vx(1)",
                "G = Pm(1)",
                "X = Vx(" + str(dimension) + ")",
                "Y = Vy(" + str(dimension) + ")",
                "P = Vy(1)", cuda_type=cuda_type))

            self.varifold_convolve.append(generic_sum(
                "Exp(-(WeightedSqDist(G, X, Y))) * Square((Nx|Ny)) * P",
                "O = Vx(1)",
                "G = Pm(1)",
                "X = Vx(" + str(dimension) + ")",
                "Y = Vy(" + str(dimension) + ")",
                "Nx = Vx(" + str(dimension) + ")",
                "Ny = Vy(" + str(dimension) + ")",
                "P = Vy(1)", cuda_type=cuda_type))

            self.gaussian_convolve_gradient_x.append(generic_sum(
                "(Px|Py) * Exp(-G*SqDist(X,Y)) * (X-Y)",
                "O = Vx(" + str(dimension) + ")",
                "G = Pm(1)",
                "X = Vx(" + str(dimension) + ")",
                "Y = Vy(" + str(dimension) + ")",
                "Px = Vx(" + str(dimension) + ")",
                "Py = Vy(" + str(dimension) + ")", cuda_type=cuda_type))

            # #   Note: the following syntax corresponds to the new upcoming Keops syntax (>v0.0.89)
            # self.gaussian_convolve.append(generic_sum(
            #     "Exp(-G*SqDist(X,Y)) * P",
            #     ["G = Pm(1)",
            #      "X = Vx(" + str(dimension) + ")",
            #      "Y = Vy(" + str(dimension) + ")",
            #      "P = Vy(" + str(dimension) + ")"],
            #     axis=1))
            #
            # self.point_cloud_convolve.append(generic_sum(
            #     "Exp(-G*SqDist(X,Y)) * P",
            #     ["G = Pm(1)",
            #      "X = Vx(" + str(dimension) + ")",
            #      "Y = Vy(" + str(dimension) + ")",
            #      "P = Vy(1)"],
            #     axis=1))
            #
            # self.varifold_convolve.append(generic_sum(
            #     "Exp(-(WeightedSqDist(G, X, Y))) * Square((Nx|Ny)) * P",
            #     ["G = Pm(1)",
            #      "X = Vx(" + str(dimension) + ")",
            #      "Y = Vy(" + str(dimension) + ")",
            #      "Nx = Vx(" + str(dimension) + ")",
            #      "Ny = Vy(" + str(dimension) + ")",
            #      "P = Vy(1)"],
            #     axis=1))
            #
            # self.gaussian_convolve_gradient_x.append(generic_sum(
            #     "(Px|Py) * Exp(-G*SqDist(X,Y)) * (X-Y)",
            #     ["G = Pm(1)",
            #      "X = Vx(" + str(dimension) + ")",
            #      "Y = Vy(" + str(dimension) + ")",
            #      "Px = Vx(" + str(dimension) + ")",
            #      "Py = Vy(" + str(dimension) + ")"],
            #     axis=1))

    def convolve(self, x, y, p, mode='gaussian'):

        if mode in ['gaussian', 'pointcloud']:
            assert x.device == y.device == p.device, 'tensors must be on the same device. x.device=' + str(x.device) \
                                                     + ', y.device=' + str(y.device) + ', p.device=' + str(p.device)

            d = x.size(1)

            device = 'GPU'
            dev, device_id = utilities.get_best_device()
            x = utilities.move_data(x, dev)
            y = utilities.move_data(y, dev)
            p = utilities.move_data(p, dev)

            # device = 'CPU'
            # dev, device_id = ('cpu', -1)

            # if x.device.type == 'cuda':
            #     device = 'GPU'
            #     device_id = x.device.index
            # else:
            #     device_id = -1
            #     device = 'CPU'
            return self.gaussian_convolve[d - 2](self.gamma.to(x.device),
                                                 x.contiguous(), y.contiguous(), p.contiguous(), backend=device, device_id=device_id).cpu()

        elif mode == 'varifold':
            assert isinstance(x, tuple), 'x must be a tuple'
            assert len(x) == 2, 'tuple length must be 2'
            assert isinstance(y, tuple), 'y must be a tuple'
            assert len(y) == 2, 'tuple length must be 2'
            assert x[0].device == y[0].device == p.device, 'x, y and p must be on the same device'
            assert x[1].device == y[1].device == p.device, 'x, y and p must be on the same device'

            x, nx = x
            y, ny = y
            d = x.size(1)

            device = 'GPU'
            dev, device_id = utilities.get_best_device()
            x = utilities.move_data(x, dev)
            nx = utilities.move_data(nx, dev)
            y = utilities.move_data(y, dev)
            ny = utilities.move_data(ny, dev)
            p = utilities.move_data(p, dev)

            # device = 'CPU'
            # dev, device_id = ('cpu', -1)

            # if x[0].device.type == 'cuda':
            #     device = 'GPU'
            #     device_id = x.device.index
            # else:
            #     device_id = -1
            #     device = 'CPU'
            return self.varifold_convolve[d - 2](self.gamma.to(x.device),
                                                 x.contiguous(), y.contiguous(), nx.contiguous(), ny.contiguous(), p.contiguous(), backend=device, device_id=device_id).cpu()

        else:
            raise RuntimeError('Unknown kernel mode.')

    def convolve_gradient(self, px, x, y=None, py=None, mode='gaussian'):
        if y is None:
            y = x
        if py is None:
            py = px

        assert px.device == x.device == y.device == py.device, 'tensors must be on the same device'

        device = 'GPU'
        dev, device_id = utilities.get_best_device()
        x = utilities.move_data(x, dev)
        px = utilities.move_data(px, dev)
        y = utilities.move_data(y, dev)
        py = utilities.move_data(py, dev)

        # device = 'CPU'
        # dev, device_id = ('cpu', -1)

        # if x.device.type == 'cuda':
        #     device = 'GPU'
        #     device_id = x.device.index
        # else:
        #     device_id = -1
        #     device = 'CPU'
        d = x.size(1)

        gamma = self.gamma.to(x.device)

        return (-2 * gamma * self.gaussian_convolve_gradient_x[d - 2](
            gamma, x, y, px, py, backend=device, device_id=device_id)).cpu()

