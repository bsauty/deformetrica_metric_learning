import logging

import torch
import torch.cuda

from support.kernels.keops_kernel import KeopsKernel

logger = logging.getLogger(__name__)


class CudaNotAvailableException(Exception):
    pass


class TorchCudaKernel(KeopsKernel):
    def __init__(self, kernel_width=None, device='cuda:0'):
        if not torch.cuda.is_available():
            raise CudaNotAvailableException

        self.device = torch.device(device)
        super().__init__(kernel_width)

    def convolve(self, x, y, p, mode=None):
        # move tensors to device if needed
        (x, y, p) = map(self.__move_tensor_to_device_if_needed, [x, y, p])
        # convolve
        res = super().convolve(x, y, p, mode)
        # wait for cuda to finish
        torch.cuda.synchronize()
        return res

    def convolve_gradient(self, px, x, y=None, py=None):
        (px, x, y, py) = map(self.__move_tensor_to_device_if_needed, [px, x, y, py])
        res = super().convolve_gradient(px, x, y, py)
        # wait for cuda to finish
        torch.cuda.synchronize()
        return res

    def __move_tensor_to_device_if_needed(self, t):
        if t is not None and t.device is not self.device:
            logger.debug('moving tensors to device: ' + str(self.device))
            res = t.to(self.device)
            assert res.device == self.device, 'error moving tensor to device'
            return res
