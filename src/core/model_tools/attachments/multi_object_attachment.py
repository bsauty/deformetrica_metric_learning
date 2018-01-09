import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../../')
from pydeformetrica.src.support.utilities.general_settings import Settings

import numpy as np
import torch
from torch.autograd import Variable


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

    def compute_weighted_distance(self, points, multi_obj1, multi_obj2, inverse_weights):
        """
        Takes two multiobjects and their new point positions to compute the distances
        """
        distances = self.compute_distances(points, multi_obj1, multi_obj2)
        assert distances.size()[0] == len(inverse_weights)
        weighted_distance = 0.
        for k in range(len(inverse_weights)): weighted_distance += distances[k] / inverse_weights[k]
        return weighted_distance

    def compute_distances(self, points, multi_obj1, multi_obj2):
        """
        Takes two multiobjects and their new point positions to compute the distances.
        """
        assert len(multi_obj1.object_list) == len(multi_obj2.object_list), \
            "Cannot compute distance between multi-objects which have different number of objects"
        distances = Variable(torch.zeros((len(multi_obj1.object_list),)).type(Settings().tensor_scalar_type),
                             requires_grad=False)

        pos = 0
        for i, obj1 in enumerate(multi_obj1.object_list):
            obj2 = multi_obj2.object_list[i]
            if self.attachment_types[i] == 'Current'.lower():
                distances[i] = self._current_distance(
                    points[pos:pos + obj1.get_number_of_points()], obj1, obj2, self.kernels[i].kernel_width)
            elif self.attachment_types[i] == 'Varifold'.lower():
                distances[i] = self._varifold_distance(
                    points[pos:pos + obj1.get_number_of_points()], obj1, obj2, self.kernels[i].kernel_width)
            elif self.attachment_types[i] == 'Landmark'.lower():
                distances[i] = self._landmark_distance(
                    points[pos:pos + obj1.get_number_of_points()], obj2)
            else:
                assert False, "Please implement the distance {e} you are trying to use :)".format(
                    e=self.attachment_types[i])
            pos += obj1.get_number_of_points()
        return distances

    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    def _current_distance(self, points, source, target, kernel):
        """
        Compute the current distance between source and target, assuming points are the new points of the source
        We assume here that the target never moves.
        """

        assert kernel.kernel_width > 0, "Please set the kernel width in OrientedSurfaceDistance computation"

        c1, n1 = source.get_centers_and_normals(points)
        c2, n2 = target.get_centers_and_normals()

        def current_scalar_product(points_1, points_2, normals_1, normals_2):
            return torch.dot(normals_1, kernel.convolve(points_1, points_2, normals_2))

        if target.norm is None:
            target.norm = current_scalar_product(c2, c2, n2, n2)

        out = current_scalar_product(c1, c1, n1, n1)
        out += target.norm
        out -= 2 * current_scalar_product(c1, c2, n1, n2)

        return out

    def _varifold_distance(self, points, source, target, kernel_width):

        """
        Returns the current distance between the 3D meshes
        source and target are SurfaceMesh objects
        points are source points (torch tensor)
        """
        c1, n1 = source.get_centers_and_normals(points)
        c2, n2 = target.get_centers_and_normals()

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

        if target.norm is None:
            target.norm = varifold_scalar_product(c2, c2, areab, areab, nbeta, nbeta)

        return varifold_scalar_product(c1, c1, areaa, areaa, nalpha, nalpha) \
               + target.norm \
               - 2 * varifold_scalar_product(c1, c2, areaa, areab, nalpha, nbeta)

    def _landmark_distance(self, points, target):
        """
        Point correspondance distance
        """
        target_points = target.get_points_torch()

        return torch.norm(points - target_points, 2) ** 2
