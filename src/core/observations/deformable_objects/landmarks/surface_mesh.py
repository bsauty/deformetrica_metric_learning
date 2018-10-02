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
                a = torch_points_coordinates[connectivity_torch[:, 0]]
                b = torch_points_coordinates[connectivity_torch[:, 1]]
                c = torch_points_coordinates[connectivity_torch[:, 2]]
                centers = (a+b+c)/3.
                self.centers = centers
                self.normals = torch.cross(b-a, c-a)/2
        else:
            a = points[connectivity_torch[:, 0]].to(device)
            b = points[connectivity_torch[:, 1]].to(device)
            c = points[connectivity_torch[:, 2]].to(device)
            centers = (a+b+c)/3.
            self.centers = centers
            self.normals = torch.cross(b-a, c-a)/2

        # assert self.centers.device == self.normals.device == torch.device(device), 'An error occurred while moving tensors to device ' + str(device)
        return self.centers.to(device), self.normals.to(device)
