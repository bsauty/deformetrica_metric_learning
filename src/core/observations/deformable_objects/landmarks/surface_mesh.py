import numpy as np
import torch
from torch.autograd import Variable

from core import default
from core.observations.deformable_objects.landmarks.landmark import Landmark


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

    # Clone.
    def clone(self):
        clone = SurfaceMesh(self.dimension)
        clone.points = np.copy(self.points)
        clone.is_modified = self.is_modified
        clone.bounding_box = self.bounding_box
        clone.norm = self.norm
        clone.connectivity = np.copy(self.connectivity)
        return clone

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def update(self):
        # self.get_centers_and_normals()
        Landmark.update(self)

    @staticmethod
    def _get_centers_and_normals(points, connectivity,
                                 tensor_scalar_type=default.tensor_scalar_type, tensor_integer_type=default.tensor_integer_type):
        if not isinstance(points, torch.Tensor):
            points = torch.from_numpy(points).type(tensor_scalar_type)
        if not isinstance(connectivity, torch.Tensor):
            connectivity = torch.from_numpy(connectivity).type(tensor_integer_type)

        a = points[connectivity[:, 0]]
        b = points[connectivity[:, 1]]
        c = points[connectivity[:, 2]]
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
                                tensor_scalar_type=default.tensor_scalar_type):
        """
        Given a new set of points, use the corresponding connectivity available in the polydata
        to compute the new normals, all in torch
        """
        connectivity_torch = torch.from_numpy(self.connectivity).type(tensor_integer_type)
        if points is None:
            if self.is_modified or self.centers is None:
                torch_points_coordinates = torch.from_numpy(self.points).type(tensor_scalar_type)
                self.centers, self.normals = SurfaceMesh._get_centers_and_normals(torch_points_coordinates, connectivity_torch,
                                                                                  tensor_scalar_type=tensor_scalar_type, tensor_integer_type=tensor_integer_type)
        else:
            self.centers, self.normals = SurfaceMesh._get_centers_and_normals(points, connectivity_torch,
                                                                              tensor_scalar_type=tensor_scalar_type, tensor_integer_type=tensor_integer_type)
        return self.centers, self.normals
