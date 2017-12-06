import os.path
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../')
import torch
from torch.autograd import Variable
from pydeformetrica.src.core.observations.deformable_objects.landmarks.landmark import Landmark


class SurfaceMesh(Landmark):
    """
    3D Triangular mesh.
    """
    def __init__(self):
        Landmark.__init__(self)
        self.connec = None#The list of cells00

    def ComputeConnectivity(self):
        self.connec = np.zeros((self.PolyData.GetNumberOfCells(), 3))
        for i in range(self.PolyData.GetNumberOfCells()):
            self.connec[i,0] = self.PolyData.GetCell(i).GetPointId(0)
            self.connec[i,1] = self.PolyData.GetCell(i).GetPointId(1)
            self.connec[i,2] = self.PolyData.GetCell(i).GetPointId(2)
        self.connec = torch.from_numpy(self.connec).type(torch.LongTensor)#Now torch tensor

    def Update(self):
        Landmark.Update(self)
        self.ComputeConnectivity()

    def GetCentersAndNormals(self, points):
        """
        Given a new set of points, use the corresponding connectivity available in the polydata
        to compute the new normals, all in torch
        """
        a,b,c = points[self.connec[:,0]], points[self.connec[:,1]], points[self.connec[:,2]]
        centers = (a+b+c)/3.
        return centers, torch.cross(b-a, c-a)
