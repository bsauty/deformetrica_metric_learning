from abc import ABC, abstractmethod


class AbstractKernel(ABC):

    def __init__(self, kernel_width=None):
        self.kernel_width = kernel_width

    @abstractmethod
    def convolve(self, x, y, p, mode=None):
        pass

    @abstractmethod
    def convolve_gradient(self, px, x, y=None, py=None):
        pass

    @abstractmethod
    def get_kernel_matrix(self, x, y=None):
        pass
