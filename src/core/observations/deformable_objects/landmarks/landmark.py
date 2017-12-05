import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../')

import numpy as np

from pydeformetrica.src.support.utilities.general_settings import GeneralSettings


class Landmark:

    """
    Landmarks (i.e. labelled point sets).
    The Landmark class represents a set of labelled points. This class assumes that the source and the target
    have the same number of points with a point-to-point correspondence.

    """

    # Constructor.
    def __init__(self):
        self.PolyData = None
        self.PointCoordinates = None
        self.IsModified = True
        self.BoundingBox = None

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

        self.IsModified = True

    # Gets the geometrical data that defines the landmark object, as a matrix list.
    def GetData(self):
        return self.PointCoordinates

    # Update the relevant information.
    def Update(self):
        if self.IsModified:
            self.UpdateBoundingBox()
            self.IsModified = False

    # Compute a tight bounding box that contains all the landmark data.
    def UpdateBoundingBox(self):
        dimension = GeneralSettings.Instance().Dimension
        self.BoundingBox = np.zeros((dimension, 2))
        for d in range(dimension):
            self.BoundingBox[d, 0] = np.min(self.PointCoordinates[:, d])
            self.BoundingBox[d, 1] = np.max(self.PointCoordinates[:, d])
