import logging
from abc import ABC, abstractmethod

import torch

logger = logging.getLogger(__name__)


class AbstractKernel(ABC):
    def __init__(self, kernel_type='undefined', kernel_width=None):
        self.kernel_type = kernel_type
        self.kernel_width = kernel_width
        logger.debug('instantiating kernel %s with kernel_width %s. addr: %s', self.kernel_type, self.kernel_width, hex(id(self)))

    def __eq__(self, other):
        return self.kernel_type == other.kernel_type and self.kernel_width == other.kernel_width

    def __hash__(self):
        return AbstractKernel.hash(self.kernel_type, self.kernel_width)

    @staticmethod
    def hash(kernel_type, kernel_width, *args, **kwargs):
        return hash((kernel_type, kernel_width))

    @abstractmethod
    def convolve(self, x, y, p, mode=None):
        raise NotImplementedError

    @abstractmethod
    def convolve_gradient(self, px, x, y=None, py=None):
        raise NotImplementedError

    def get_kernel_matrix(self, x, y=None):
        """
        returns the kernel matrix, A_{ij} = exp(-|x_i-x_j|^2/sigma^2)
        """
        if y is None:
            y = x
        assert (x.size(0) == y.size(0))
        sq = self._squared_distances(x, y)
        return torch.exp(-sq / (self.kernel_width ** 2))

    @staticmethod
    def _squared_distances(x, y):
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
        return dist
