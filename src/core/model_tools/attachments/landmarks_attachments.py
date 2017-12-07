import numpy as np
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../../')
from pydeformetrica.src.support.utilities.torch_kernel import TorchKernel


#This file contains functions computing, in torch, the distances between meshes

def CurrentDistance(points, source, target, kernel_width=0.):
    """
    Compute the current distance between source and target, assuming points are the new points of the source
    We assume here that the target never move.
    """

    assert kernel_width > 0, "Please set the kernel width in OrientedSurfaceDistance computation"

    kernel = TorchKernel()
    kernel.KernelWidth = kernel_width
    c1, n1 = source.GetCentersAndNormals(points)
    c2, n2 = target.GetCentersAndNormals()

    def current_scalar_product(p1, p2, n1, n2):
        return torch.dot(n1.view(-1), kernel.Convolve(p1,n2,p2).view(-1))

    if target.Norm is None:
        target.Norm = current_scalar_product(c2, c2, n2, n2)
    out = current_scalar_product(c1, c1, n1, n1)
    out += target.Norm
    out -= 2*current_scalar_product(c1, c2, n1, n2)

    return out


def NonOrientedSurfaceDistance(points1, surf1, surf2, kernel_width=0.):
    """
    Returns the current distance between the 3D meshes
    surf1 and surf2 are SurfaceMesh objects
    points1 and points2 are tensors for the points
    It uses the connectivity matrices of surf1 and
    surf2 (via GetCentersAndNormals) to compute centers
    and normals given the new points
    """
    pass
    # assert kernel_width>0, "Please set the kernel width in OrientedSurfaceDistance computation"
    # kernel = TorchKernel()
    # kernel.KernelWidth = kernel_width
    # c1, n1 = surf1.GetCentersAndNormals(points1)
    # c2, n2 = surf2.GetCentersAndNormals()
    # def current_scalar_product(p1, p2, n1, n2):
    #     return torch.dot(n1.view(-1), kernel.Convolve(p1,n2,p2).view(-1))
    # out = current_scalar_product(c1, c1, n1, n1)
    # out += current_scalar_product(c2, c2, n2, n2)
    # out -= 2*current_scalar_product(c1, c2, n1, n2)
    # return out
