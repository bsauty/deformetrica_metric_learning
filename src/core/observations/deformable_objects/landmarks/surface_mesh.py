import numpy as np
import torch
from torch.autograd import Variable

from core import default
from core.observations.deformable_objects.landmarks.landmark import Landmark
from support import utilities


class SurfaceMesh(Landmark):
    """
    3D Triangular mesh.
    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, dimension):
        Landmark.__init__(self, dimension)
        self.type = 'SurfaceMesh'

        self.connectivity = None
        # All of these are torch tensor attributes.
        self.centers = None
        self.normals = None

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def update(self):
        # self.get_centers_and_normals()
        Landmark.update(self)

    @staticmethod
    def _get_centers_and_normals(points, connectivity,
                                 tensor_scalar_type=default.tensor_scalar_type, tensor_integer_type=default.tensor_integer_type,
                                 device='cpu'):
        if not isinstance(points, torch.Tensor):
            points = torch.from_numpy(points).type(tensor_scalar_type)
        if not isinstance(connectivity, torch.Tensor):
            connectivity = torch.from_numpy(connectivity).type(tensor_integer_type)

        a = points[connectivity[:, 0]].to(device)
        b = points[connectivity[:, 1]].to(device)
        c = points[connectivity[:, 2]].to(device)
        centers = (a + b + c) / 3.
        normals = torch.cross(b - a, c - a) / 2
        return centers, normals

    @staticmethod
    def check_for_null_normals(normals):
        """
        Check to see if given tensor contains zeros.
        cf: https://discuss.pytorch.org/t/find-indices-with-value-zeros/10151
        :param normals: input tensor
        :return:  True if normals does not contain zeros
                  False if normals contains zeros
        """
        return (torch.norm(normals, 2, 1) == 0).nonzero().sum() == 0

    def get_centers_and_normals(self, points=None,
                                tensor_integer_type=default.tensor_integer_type,
                                tensor_scalar_type=default.tensor_scalar_type,
                                device='cpu'):
        """
        Given a new set of points, use the corresponding connectivity available in the polydata
        to compute the new normals, all in torch
        """
        connectivity_torch = utilities.move_data(self.connectivity, device=device, dtype=tensor_integer_type)
        if points is None:
            if self.is_modified or self.centers is None:
                torch_points_coordinates = utilities.move_data(self.points, device=device, dtype=tensor_scalar_type)
                self.centers, self.normals = SurfaceMesh._get_centers_and_normals(torch_points_coordinates, connectivity_torch,
                                                                                  tensor_scalar_type=tensor_scalar_type, tensor_integer_type=tensor_integer_type,
                                                                                  device=device)
        else:
            self.centers, self.normals = SurfaceMesh._get_centers_and_normals(points, connectivity_torch,
                                                                              tensor_scalar_type=tensor_scalar_type, tensor_integer_type=tensor_integer_type,
                                                                              device=device)
        return self.centers.to(device), self.normals.to(device)
