import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../')

from pydeformetrica.src.core.observations.deformable_objects.landmarks.landmark import Landmark
from pydeformetrica.src.support.utilities.general_settings import GeneralSettings

import numpy as np

class SurfaceMesh(Landmark):

    """
    Triangular mesh.

    """

    def SetPolyData(self, polyData):
        self.PolyData = polyData

        numberOfPoints = polyData.GetNumberOfPoints()
        dimension = GeneralSettings.Instance().Dimension
        pointCoordinates = np.zeros((numberOfPoints, dimension))

        for k in range(numberOfPoints):
            p = polyData.GetPoint(k)
            pointCoordinates[k,:] = p[0:dimension]
        self.PointCoordinates = pointCoordinates

        self.SetModified()

    def SetModified(self):
        self.IsModified = True




