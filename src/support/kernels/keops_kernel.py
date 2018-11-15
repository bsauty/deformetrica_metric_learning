import torch

from support import utilities
from support.kernels import AbstractKernel
from pykeops.torch import Genred
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
            self.gaussian_convolve.append(Genred(
                "Exp(-G*SqDist(X,Y)) * P",
                ["G = Pm(1)",
                 "X = Vx(" + str(dimension) + ")",
                 "Y = Vy(" + str(dimension) + ")",
                 "P = Vy(" + str(dimension) + ")"],
                reduction_op='Sum', axis=1, cuda_type=cuda_type))

            self.point_cloud_convolve.append(Genred(
                "Exp(-G*SqDist(X,Y)) * P",
                ["G = Pm(1)",
                 "X = Vx(" + str(dimension) + ")",
                 "Y = Vy(" + str(dimension) + ")",
                 "P = Vy(1)"],
                reduction_op='Sum', axis=1, cuda_type=cuda_type))

            self.varifold_convolve.append(Genred(
                "Exp(-(WeightedSqDist(G, X, Y))) * Square((Nx|Ny)) * P",
                ["G = Pm(1)",
                 "X = Vx(" + str(dimension) + ")",
                 "Y = Vy(" + str(dimension) + ")",
                 "Nx = Vx(" + str(dimension) + ")",
                 "Ny = Vy(" + str(dimension) + ")",
                 "P = Vy(1)"],
                reduction_op='Sum', axis=1, cuda_type=cuda_type))

            self.gaussian_convolve_gradient_x.append(Genred(
                "(Px|Py) * Exp(-G*SqDist(X,Y)) * (X-Y)",
                ["G = Pm(1)",
                 "X = Vx(" + str(dimension) + ")",
                 "Y = Vy(" + str(dimension) + ")",
                 "Px = Vx(" + str(dimension) + ")",
                 "Py = Vy(" + str(dimension) + ")"],
                reduction_op='Sum', axis=1, cuda_type=cuda_type))

    def convolve(self, x, y, p, mode='gaussian', return_to_cpu=True):
        if mode in ['gaussian', 'pointcloud']:
            assert isinstance(x, torch.Tensor), 'x variable must be a torch Tensor'
            assert isinstance(y, torch.Tensor), 'y variable must be a torch Tensor'
            assert isinstance(p, torch.Tensor), 'p variable must be a torch Tensor'
            assert x.device == y.device == p.device, 'tensors must be on the same device. x.device=' + str(x.device) \
                                                     + ', y.device=' + str(y.device) + ', p.device=' + str(p.device)

            d = x.size(1)
            gamma = self.gamma.to(x.device)

            device_id = x.device.index if x.device.index is not None else -1
            res = self.gaussian_convolve[d - 2](gamma, x.contiguous(), y.contiguous(), p.contiguous(), device_id=device_id)
            return res.cpu() if return_to_cpu else res

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
            gamma = self.gamma.to(x.device)

            device_id = x.device.index if x.device.index is not None else -1
            res = self.varifold_convolve[d - 2](gamma, x.contiguous(), y.contiguous(), nx.contiguous(), ny.contiguous(), p.contiguous(), device_id=device_id)
            return res.cpu() if return_to_cpu else res

        else:
            raise RuntimeError('Unknown kernel mode.')

    def convolve_gradient(self, px, x, y=None, py=None, mode='gaussian', device='cpu', return_to_cpu=True):
        if y is None:
            y = x
        if py is None:
            py = px

        assert isinstance(px, torch.Tensor), 'px variable must be a torch Tensor'
        assert isinstance(x, torch.Tensor), 'x variable must be a torch Tensor'
        assert isinstance(y, torch.Tensor), 'y variable must be a torch Tensor'
        assert isinstance(py, torch.Tensor), 'py variable must be a torch Tensor'
        assert px.device == x.device == y.device == py.device, 'tensors must be on the same device'

        d = x.size(1)
        gamma = self.gamma.to(x.device)

        device_id = x.device.index if x.device.index is not None else -1
        res = (-2 * gamma * self.gaussian_convolve_gradient_x[d - 2](gamma, x, y, px, py, device_id=device_id))
        return res.cpu() if return_to_cpu else res
