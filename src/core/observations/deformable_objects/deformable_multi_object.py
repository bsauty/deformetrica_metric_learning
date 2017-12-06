import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

import numpy as np

from pydeformetrica.src.core.observations.deformable_objects.landmarks.landmark import Landmark
from pydeformetrica.src.support.linear_algebra.matrix_list import MatrixList
from pydeformetrica.src.support.utilities.general_settings import GeneralSettings


class DeformableMultiObject:

    """
    Collection of deformable objects, i.e. landmarks or images.
    The DeformableMultiObject class is used to deal with collections of deformable objects embedded in
    the current 2D or 3D space. It extends for such collections the methods of the deformable object class
    that are designed for a single object at a time.

    """

    # Constructor.
    def __init__(self):
        self.ObjectList = []
        self.NumberOfObjects = None
        self.BoundingBox = None

    # Accessor
    def __getitem__(self, item):
        return self.ObjectList[item]

    # Update the relevant information.
    def Update(self):
        self.NumberOfObjects = len(self.ObjectList)
        assert(self.NumberOfObjects > 0)

        for k in range(self.NumberOfObjects):
            self.ObjectList[k].Update()

        self.UpdateBoundingBox()

    # Compute a tight bounding box that contains all objects.
    def UpdateBoundingBox(self):
        assert(self.NumberOfObjects > 0)
        dimension = GeneralSettings.Instance().Dimension

        self.BoundingBox = self.ObjectList[0].BoundingBox
        for k in range(1, self.NumberOfObjects):
            for d in range(dimension):
                if self.ObjectList[k].BoundingBox[d, 0] < self.BoundingBox[d, 0]:
                    self.BoundingBox[d, 0] = self.ObjectList[k].BoundingBox[d, 0]
                if self.ObjectList[k].BoundingBox[d, 1] > self.BoundingBox[d, 1]:
                    self.BoundingBox[d, 1] = self.ObjectList[k].BoundingBox[d, 1]

    # Gets the geometrical data that defines the deformable multi object, as a matrix list.
    def GetData(self):
        out = MatrixList()
        for k in range(len(self.ObjectList)):
            out.append(self.ObjectList[k].GetData())
        return out
