import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../')

from pydeformetrica.src.core.observations.deformable_objects.landmarks.landmark import Landmark


class SurfaceMesh(Landmark):
    """
    3D Triangular mesh.
    """
    def __init__(self):
        super(Landmark, self).__init__()
        self.connec = None#The list of cells00

    def ComputeConnectivity(self):
        if self.connect == None:
            self.connec = np.zeros((self.polyData.GetNumberOfCells()))
            for i in range(polyData.GetNumberOfCells()):
                self.connec[i,0] = self.polyData.GetCell(i).GetPointId(0)
                self.connec[i,1] = self.polyData.GetCell(i).GetPointId(1)
                self.connec[i,2] = self.polyData.GetCell(i).GetPointId(2)
            self.connec = Variable(torch.from_numpy(self.connec))#Now torch tensor

    def GetCentersAndNormals(self, points):
        """
        Given a new set of points, use the corresponding connectivity available in the polydata
        to compute the new normals, all in torch
        """
        #We have the connectivity, so should look like
        #normal[i] = cross_3D(points[connec[i,1]] - points[connec[i,0]], points[connec[i,2]] - points[connec[i,0]])
        #How to to this in torch ?
        a,b,c = points[self.connect[:,0]], points[self.connect[:,1]], points[self.connect[:,2]]
        centers = (a+b+c)/3.
        edges = torch.stack([a,b,c])
        return centers, torch.cross(edges)
