import os.path
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../')
import torch
from torch.autograd import Variable

from pydeformetrica.src.core.observations.deformable_objects.landmarks.landmark import Landmark
from pydeformetrica.src.support.utilities.general_settings import Settings


class SurfaceMesh(Landmark):
    """
    3D Triangular mesh.
    """

    ####################################################################################################################
    ### Constructors:
    ####################################################################################################################

    def __init__(self):
        Landmark.__init__(self)
        self.connectivity = None #The list of cells
        self.centers = None
        self.normals = None

    def clone(self):
        clone = SurfaceMesh()

        # Superclass attributes.
        clone.poly_data = self.poly_data
        clone.point_coordinates = self.point_coordinates
        clone.is_modified = self.is_modified
        clone.bounding_box = self.bounding_box
        clone.norm = self.norm

        # Own atributes.
        clone.connectivity = self.connectivity
        clone.centers = self.centers
        clone.normals = self.normals

        return clone

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def update(self):
        Landmark.update(self)
        self.compute_connectivity()
        self.get_centers_and_normals()

    def compute_connectivity(self):
        self.connectivity = np.zeros((self.poly_data.GetNumberOfCells(), 3))
        for i in range(self.poly_data.GetNumberOfCells()):
            self.connectivity[i, 0] = self.poly_data.GetCell(i).GetPointId(0)
            self.connectivity[i, 1] = self.poly_data.GetCell(i).GetPointId(1)
            self.connectivity[i, 2] = self.poly_data.GetCell(i).GetPointId(2)
        self.connectivity = torch.from_numpy(self.connectivity).type(Settings().tensor_integer_type)

    def get_centers_and_normals(self, points=None):
        """
        Given a new set of points, use the corresponding connectivity available in the polydata
        to compute the new normals, all in torch
        """
        if points is None:
            if (self.normals is None) or (self.centers is None):
                torch_points_coordinates = Variable(
                    torch.from_numpy(self.point_coordinates).type(Settings().tensor_scalar_type))
                a, b, c = torch_points_coordinates[self.connectivity[:, 0]], \
                          torch_points_coordinates[self.connectivity[:, 1]], \
                          torch_points_coordinates[self.connectivity[:, 2]]
                centers = (a+b+c)/3.
                self.centers = centers
                self.normals = torch.cross(b-a, c-a)
        else:
            a, b, c = points[self.connectivity[:, 0]], points[self.connectivity[:, 1]], points[self.connectivity[:, 2]]
            centers = (a+b+c)/3.
            self.centers = centers
            self.normals = torch.cross(b-a, c-a)
        return self.centers, self.normals
