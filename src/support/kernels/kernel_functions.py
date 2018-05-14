import warnings

import torch

from support.kernels.exact_kernel import ExactKernel

if torch.cuda.is_available():
    from support.kernels.cuda_exact_kernel import CudaExactKernel


# Creates a longitudinal dataset object from xml parameters.
def create_kernel(kernel_type, kernel_width):

    if kernel_type.lower() == 'exact':
        kernel = ExactKernel()

    elif kernel_type.lower() == 'no_kernel_needed':
        return None

    elif kernel_type.lower() == 'cudaexact':
        if not (torch.cuda.is_available()):
            kernel = ExactKernel()
            msg = 'Cuda seems to be unavailable. Overriding the "CudaExact" kernel type with "Exact" type.'
            warnings.warn(msg)

        else:
            kernel = CudaExactKernel()

    else:
        kernel = ExactKernel()
        msg = 'Unknown kernel type: "' + kernel_type + '". Defaulting to "ExactKernel"'
        warnings.warn(msg)

    kernel.kernel_width = kernel_width
    return kernel
