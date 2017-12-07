import os.path
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../')
import torch
from torch.autograd import Variable

from pydeformetrica.src.core.observations.deformable_objects.landmarks.landmark import Landmark
from pydeformetrica.src.support.utilities.general_settings import *


class SurfaceMesh(Landmark):
    """
    3D Triangular mesh.
    """
    def __init__(self):
        Landmark.__init__(self)
        self.connec = None #The list of cells
        self.centers = None
        self.normals = None

    def ComputeConnectivity(self):
        self.connec = np.zeros((self.PolyData.GetNumberOfCells(), 3))
        for i in range(self.PolyData.GetNumberOfCells()):
            self.connec[i,0] = self.PolyData.GetCell(i).GetPointId(0)
            self.connec[i,1] = self.PolyData.GetCell(i).GetPointId(1)
            self.connec[i,2] = self.PolyData.GetCell(i).GetPointId(2)
        self.connec = torch.from_numpy(self.connec).type(torch.LongTensor) #Now torch tensor

    def Update(self):
        Landmark.Update(self)
        self.ComputeConnectivity()
        self.GetCentersAndNormals()

    def GetCentersAndNormals(self, points=None):
        """
        Given a new set of points, use the corresponding connectivity available in the polydata
        to compute the new normals, all in torch
        """
        if points is None:
            if (self.normals is None) or (self.centers is None):
                torchPointsCoordinates = Variable(torch.from_numpy(self.PointCoordinates).type(Settings().TensorType))
                a,b,c = torchPointsCoordinates[self.connec[:,0]], torchPointsCoordinates[self.connec[:,1]], torchPointsCoordinates[self.connec[:,2]]
                centers = (a+b+c)/3.
                self.centers = centers
                self.normals = torch.cross(b-a, c-a)
        else:
            a,b,c = points[self.connec[:,0]], points[self.connec[:,1]], points[self.connec[:,2]]
            centers = (a+b+c)/3.
            self.centers = centers
            self.normals = torch.cross(b-a, c-a)
        return self.centers, self.normals
