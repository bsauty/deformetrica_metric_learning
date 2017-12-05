import numpy as np
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../../')
from pydeformetrica.src.support.utilities.torch_kernel import TorchKernel


#This file contains functions computing, in torch, the distances between meshes

def OrientedSurfaceDistance(points1, points2, surf1, surf2, kernel_width=0.):
    """
    Returns the current distance between the 3D meshes
    surf1 and surf2 are SurfaceMesh objects
    points1 and points2 are tensors for the points
    It uses the connectivity matrices of surf1 and
    surf2 (via GetCentersAndNormals) to compute centers
    and normals given the new points
    """
    assert kernel_width>0, "Please set the kernel width in OrientedSurfaceDistance computation"
    kernel = TorchKernel()
    kernel.KernelWidth = kernel_width
    c1, n1 = surf1.GetCentersAndNormals(points1)
    c2, n2 = surf2.GetCentersAndNormals(points2)
    print(c2, n2)
    def current_scalar_product(p1, p2, n1, n2):
        return torch.dot(n1.view(-1), kernel.Convolve(p1,n2,p2).view(-1))
    out = current_scalar_product(c1, c1, n1, n1)
    out += current_scalar_product(c2, c2, n2, n2)
    out -= 2*current_scalar_product(c1, c2, n1, n2)
    return out
