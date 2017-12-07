import numpy as np
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../../')
from pydeformetrica.src.support.utilities.torch_kernel import TorchKernel


#This file contains functions computing, in torch, the distances between meshes

def CurrentDistance(points1, surf1, surf2, kernel_width=0.):
    """
    Returns the current distance between the 3D meshes
    surf1 and surf2 are SurfaceMesh objects
    points1 and points2 are tensors for the points
    It uses the connectivity matrices of surf1 and
    surf2 (via GetCentersAndNormals) to compute centers
    and normals given the new points

    """

    assert kernel_width > 0, "Please set the kernel width in OrientedSurfaceDistance computation"

    kernel = TorchKernel()
    kernel.KernelWidth = kernel_width
    c1, n1 = surf1.GetCentersAndNormals(points1)
    c2, n2 = surf2.GetCentersAndNormals()

    def current_scalar_product(p1, p2, n1, n2):
        return torch.dot(n1.view(-1), kernel.Convolve(p1, n2, p2).view(-1))

    out = current_scalar_product(c1, c1, n1, n1)
    out += current_scalar_product(c2, c2, n2, n2)
    out -= 2*current_scalar_product(c1, c2, n1, n2)

    return out


def VarifoldDistance(points1, surf1, surf2, kernel_width):
    """
    Returns the current distance between the 3D meshes
    surf1 and surf2 are SurfaceMesh objects
    points1 and points2 are tensors for the points
    It uses the connectivity matrices of surf1 and
    surf2 (via GetCentersAndNormals) to compute centers
    and normals given the new points
    """
    c1, n1 = surf1.GetCentersAndNormals(points1)
    c2, n2 = surf2.GetCentersAndNormals()

    # alpha = normales non unitaires
    areaa = torch.norm(n1, axis=1)
    areab = torch.norm(n2, axis=1)

    nalpha = n1 / areaa.unsqueeze(1)
    nbeta = n2 / areab.unsqueeze(1)

    def gaussian(r2, s):
        return torch.exp(-r2/(s*s))
    def binet(prs):
        return prs ** 2
    def squdistance_matrix(ax, by):
        return np.sum((ax[:, np.newaxis, :] - by[np.newaxis, :, :]) ** 2, axis=2)

    def varifold_scalar_product(x, y, areaa, areab, nalpha, nbeta):
        return torch.sum(
            (areaa.unsqueeze(1) * areab.unsqueeze(0)) * gaussian(squdistance_matrix(x, y), kernel_width)
            * binet(torch.mm(nalpha, nbeta.T)), axis=1)

    return varifold_scalar_product(c1, c1, areaa, areaa, nalpha, nalpha) \
           + varifold_scalar_product(c2, c2, areab, areab, nbeta, nbeta) \
           - 2 * varifold_scalar_product(c1, c2, areaa, areab, nalpha, nbeta)



