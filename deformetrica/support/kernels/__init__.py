from enum import Enum, auto

from ...core import default
from ...support.kernels.abstract_kernel import AbstractKernel
from .keops_kernel import test_keops_setup


class Type(Enum):
    from ...support.kernels.torch_kernel import TorchKernel
    from ...support.kernels.keops_kernel import KeopsKernel

    AUTO = auto()
    NO_KERNEL = auto()
    TORCH = TorchKernel
    KEOPS = KeopsKernel


instance_map = dict()


def kernel_selector():
    # check that keops is properly setup
    res = Type.KEOPS if test_keops_setup() else Type.TORCH

    # # estimate tensor size in bytes TODO
    # import torch
    # from ..utilities import estimate_tensor_required_size
    # t = torch.tensor([[63.], [90.]], dtype=torch.float64, requires_grad=False)
    # tensor_size = estimate_tensor_required_size(t)

    return res


def factory(kernel_type, cuda_type=None, gpu_mode=None, *args, **kwargs):
    """Return an instance of a kernel corresponding to the requested kernel_type"""
    if cuda_type is None:
        cuda_type = default.dtype
    if gpu_mode is None:
        gpu_mode = default.gpu_mode

    # turn enum string to enum object
    if isinstance(kernel_type, str):
        try:
            for c in [' ', '-']:    # chars to be replaced for normalization
                kernel_type = kernel_type.replace(c, '_')
            kernel_type = Type[kernel_type.upper()]
        except:
            raise TypeError('kernel_type ' + kernel_type + ' could not be found')

    if not isinstance(kernel_type, Type):
        raise TypeError('kernel_type must be an instance of KernelType Enum')

    if kernel_type in [Type.NO_KERNEL]:
        return None

    if kernel_type in [Type.AUTO]:
        kernel_type = kernel_selector()

    res = None
    hash = AbstractKernel.hash(kernel_type, cuda_type, gpu_mode, *args, **kwargs)
    if hash not in instance_map:
        res = kernel_type.value(gpu_mode=gpu_mode, cuda_type=cuda_type, *args, **kwargs)    # instantiate
        instance_map[hash] = res
    else:
        res = instance_map[hash]

    assert res is not None
    return res
