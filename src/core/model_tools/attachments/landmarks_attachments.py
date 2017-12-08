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
        return torch.dot(n1.view(-1), kernel.Convolve(p1, n2, p2).view(-1))

    if target.Norm is None:
        target.Norm = current_scalar_product(c2, c2, n2, n2)
    out = current_scalar_product(c1, c1, n1, n1)
    out += target.Norm
    out -= 2*current_scalar_product(c1, c2, n1, n2)

    return out


def VarifoldDistance(points, source, target, kernel_width):

    """
    Returns the current distance between the 3D meshes
    source and target are SurfaceMesh objects
    points1 are source points (torch)
    """
    c1, n1 = source.GetCentersAndNormals(points)
    c2, n2 = target.GetCentersAndNormals()

    # alpha = normales non unitaires
    areaa = torch.norm(n1, 2, 1)
    areab = torch.norm(n2, 2, 1)

    nalpha = n1 / areaa.unsqueeze(1)
    nbeta = n2 / areab.unsqueeze(1)

    def gaussian(r2, s):
        return torch.exp(-r2/(s*s))

    def binet(prs):
        return prs ** 2

    def squdistance_matrix(ax, by):
        return torch.sum((ax.unsqueeze(1) - by.unsqueeze(0)) ** 2, 2)

    def varifold_scalar_product(x, y, areaa, areab, nalpha, nbeta):
        return torch.sum(torch.sum(
            areaa.unsqueeze(1) * areab.unsqueeze(0)
            * gaussian(squdistance_matrix(x, y)
            * binet(torch.mm(nalpha, torch.t(nbeta))), kernel_width), 1), 0)

    if target.Norm is None:
        target.Norm = varifold_scalar_product(c2, c2, areab, areab, nbeta, nbeta)

    return varifold_scalar_product(c1, c1, areaa, areaa, nalpha, nalpha) \
           + target.Norm \
           - 2 * varifold_scalar_product(c1, c2, areaa, areab, nalpha, nbeta)
