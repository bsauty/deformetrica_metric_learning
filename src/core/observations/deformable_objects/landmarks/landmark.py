import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../')

from pydeformetrica.src.support.utilities.general_settings import GeneralSettings

import vtk
import numpy

class Landmark:

    """
    Landmarks (i.e. labelled point sets).
    The Landmark class represents a set of labelled points. This class assumes that the source and the target
    have the same number of points with a point-to-point correspondence.

    """

    def SetAnatomicalCoordinateSystem(self, anatomicalCoordinateSystem):
        self.AnatomicalCoordinateSystem = anatomicalCoordinateSystem

    def SetPolyData(self, pointSet):
        self.PointSet = pointSet
        self.NumberOfPoints = pointSet.GetNumberOfPoints()

        dimension = GeneralSettings.Instance().AmbientSpaceDimension
        self.PointCoordinates = numpy.array((self.NumberOfPoints, dimension))

        for i in range(self.NumberOfPoints):
            p = self.PointSet.GetPoint(i)
            for dim in range(dimension):
                self.PointCoordinates(i, dim) = p(dim)

