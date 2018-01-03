import os.path
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../')
import torch
from torch.autograd import Variable

from pydeformetrica.src.core.observations.deformable_objects.landmarks.landmark import Landmark
from pydeformetrica.src.support.utilities.general_settings import Settings


class PointCloud(Landmark):
    """
    Points in 2D or 3D space, seen as measures
    """
    def __init__(self):
        Landmark.__init__(self)
        self.centers = None
        self.normals = None#This is going to be point weights, uniform for now TODO: read somewhere e.g. in the vtk the weights of the points.

    def update(self):
        Landmark.update(self)
        self.get_centers_and_normals()

    def get_centers_and_normals(self, points=None):
        """
        Given a new set of points, use the corresponding connectivity available in the polydata
        to compute the new normals (which are tangents in this case) and centers
        """
        if points is None:
            if (self.normals is None) or (self.centers is None):
                self.centers = Variable(torch.from_numpy(self.points).type(Settings().tensor_scalar_type))
                self.normals = Variable(torch.from_numpy(np.array([[1./len(self.points)] for _ in self.points])).type(Settings().tensor_scalar_type))
        else:
            self.centers = points
            self.normals = Variable(torch.from_numpy(np.array([[1./len(points)] for _ in points])).type(Settings().tensor_scalar_type))

        return self.centers, self.normals
