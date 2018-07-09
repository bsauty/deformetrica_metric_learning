from abc import ABC, abstractmethod
import torch

from support.utilities.general_settings import Settings


class AbstractKernel(ABC):
    def __init__(self, kernel_width=None):
        self.kernel_width = kernel_width

    @abstractmethod
    def convolve(self, x, y, p, mode=None):
        pass

    @abstractmethod
    def convolve_gradient(self, px, x, y=None, py=None):
        pass

    def get_kernel_matrix(self, x, y=None):
        """
        returns the kernel matrix, A_{ij} = exp(-|x_i-x_j|^2/sigma^2)
        """
        if y is None: y = x
        assert (x.size()[0] == y.size()[0])
        sq = self._squared_distances(x, y)
        return torch.exp(-sq / (self.kernel_width ** 2))

    def _squared_distances(self, x, y):
        """
        Returns the matrix of $(x_i - y_j)^2$.
        Output is of size (1, M, N).
        """
        return torch.sum((x.unsqueeze(1) - y.unsqueeze(0)) ** 2, 2)

    @staticmethod
    def _check_tensor_device(device, *args):
        def check_tensor(tensor):
            if device != tensor.device:
                raise TypeError('tensors are not all on the same device')

        for d in args:
            is_tuple = isinstance(d, tuple)
            if is_tuple:
                # if device is None:
                #     device = d[0].device
                for t in d:
                    check_tensor(t)
            else:
                # if device is None:
                #     device = d.device
                check_tensor(d)
