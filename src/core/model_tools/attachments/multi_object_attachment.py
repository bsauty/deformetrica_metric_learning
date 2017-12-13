import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../../')

import numpy as np
import torch

from pydeformetrica.src.core.model_tools.attachments.landmarks_attachments import *

class MultiObjectAttachment:

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self):
        # List of strings, e.g. 'varifold' or 'current'.
        self.attachment_types = []

        # List of kernel objects.
        self.kernels = []

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def compute_weighted_distance(self, points1, multi_obj1, multi_obj2, weights, objectNorms):
        """
        Takes two multiobjects and their new point positions to compute the distances
        """
        weighted_distance = 0.
        pos = 0
        assert len(multi_obj1.ObjectList) == len(multi_obj2.ObjectList), "Cannot compute distance between multi-objects which have different number of objects"
        for i, obj1 in enumerate(multi_obj1.ObjectList):
            obj2 = multi_obj2.ObjectList[i]
            if objectNorms[i] == 'Current':
                weighted_distance += weights[i] * self._current_distance(
                    points1[pos:pos+obj1.GetNumberOfPoints()], obj1, obj2, kernel=self.kernels[i])
            elif objectNorms[i] == 'Varifold':
                weighted_distance += weights[i] * self._varifold_distance(
                    points1[pos:pos + obj1.GetNumberOfPoints()], obj1, obj2, kernel_width=self.kernels[i].kernel_width)
            else:
                assert False, "Please implement the distance you are trying to use :)"
            pos += obj1.GetNumberOfPoints()
        return weighted_distance


    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    def _current_distance(points, source, target, kernel):
        """
        Compute the current distance between source and target, assuming points are the new points of the source
        We assume here that the target never move.
        """

        assert kernel.kernel_width > 0, "Please set the kernel width in OrientedSurfaceDistance computation"

        c1, n1 = source.GetCentersAndNormals(points)
        c2, n2 = target.GetCentersAndNormals()

        def current_scalar_product(p1, p2, n1, n2):
            return torch.dot(n1.view(-1), kernel.Convolve(p1, p2, n2).view(-1))

        if target.Norm is None:
            target.Norm = current_scalar_product(c2, c2, n2, n2)
        out = current_scalar_product(c1, c1, n1, n1)
        out += target.Norm
        out -= 2 * current_scalar_product(c1, c2, n1, n2)

        return out

    def _varifold_distance(points, source, target, kernel_width):

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
            return torch.exp(-r2 / (s * s))

        def binet(prs):
            return prs ** 2

        def squdistance_matrix(ax, by):
            return torch.sum((ax.unsqueeze(1) - by.unsqueeze(0)) ** 2, 2)

        def varifold_scalar_product(x, y, areaa, areab, nalpha, nbeta):
            return torch.sum(torch.sum(
                areaa.unsqueeze(1) * areab.unsqueeze(0)
                * gaussian(squdistance_matrix(x, y), kernel_width)
                * binet(torch.mm(nalpha, torch.t(nbeta))), 1), 0)

        if target.Norm is None:
            target.Norm = varifold_scalar_product(c2, c2, areab, areab, nbeta, nbeta)

        return varifold_scalar_product(c1, c1, areaa, areaa, nalpha, nalpha) \
               + target.Norm \
               - 2 * varifold_scalar_product(c1, c2, areaa, areab, nalpha, nbeta)