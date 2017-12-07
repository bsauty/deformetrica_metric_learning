import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

import numpy as np

from pydeformetrica.src.core.observations.deformable_objects.landmarks.landmark import Landmark
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

        for elt in self.ObjectList:
            elt.Update()

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

    # Gets the geometrical data that defines the deformable multi object, as a concatenated array.
    # We suppose that object data share the same last dimension (e.g. the list of list of points of vtks).
    def GetData(self):
        return np.concatenate([elt.GetData() for elt in self.ObjectList])

    def SetData(self, points):
        """
        points is a numpy array containing the new position of all the landmark points
        """
        assert len(points) == np.sum([elt.GetNumberOfPoints() for elt in self.ObjectList]), "Number of points differ in template and data given to template"
        pos = 0
        for i,elt in enumerate(self.ObjectList):
            elt.SetPoints(points[pos:pos+elt.GetNumberOfPoints()])
            pos += elt.GetNumberOfPoints()

    def Write(self, names):
        """
        Save the list of objects with the given names
        """
        assert len(names) == len(self.ObjectList), "Give as many names as objects to save multi-object"
        for i, name in enumerate(names):
            self.ObjectList[i].Write(name)
