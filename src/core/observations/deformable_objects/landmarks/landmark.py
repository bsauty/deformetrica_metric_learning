import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../')

from pydeformetrica.src.support.utilities.general_settings import GeneralSettings

import numpy as np


class Landmark:

    """
    Landmarks (i.e. labelled point sets).
    The Landmark class represents a set of labelled points. This class assumes that the source and the target
    have the same number of points with a point-to-point correspondence.

    """

    # Sets the PolyData attribute, and initializes the PointCoordinates one according to the ambient space dimension.
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

    # Sets the IsModified flag to true.
    def SetModified(self):
        self.IsModified = True

