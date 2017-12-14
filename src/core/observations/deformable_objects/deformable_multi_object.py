import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

import numpy as np

from pydeformetrica.src.core.observations.deformable_objects.landmarks.landmark import Landmark
from pydeformetrica.src.support.utilities.general_settings import Settings


class DeformableMultiObject:

    """
    Collection of deformable objects, i.e. landmarks or images.
    The DeformableMultiObject class is used to deal with collections of deformable objects embedded in
    the current 2D or 3D space. It extends for such collections the methods of the deformable object class
    that are designed for a single object at a time.

    """

    # Constructor.
    def __init__(self):
        self.object_list = []
        self.number_of_objects = None
        self.bounding_box = None

    # Accessor
    def __getitem__(self, item):
        return self.object_list[item]

    # Update the relevant information.
    def update(self):
        self.number_of_objects = len(self.object_list)
        assert(self.number_of_objects > 0)

        for elt in self.object_list:
            elt.update()

        self.update_bounding_box()

    # Compute a tight bounding box that contains all objects.
    def update_bounding_box(self):
        assert(self.number_of_objects > 0)
        dimension = Settings().Dimension

        self.bounding_box = self.object_list[0].bounding_box
        for k in range(1, self.number_of_objects):
            for d in range(dimension):
                if self.object_list[k].bounding_box[d, 0] < self.bounding_box[d, 0]:
                    self.bounding_box[d, 0] = self.object_list[k].bounding_box[d, 0]
                if self.object_list[k].bounding_box[d, 1] > self.bounding_box[d, 1]:
                    self.bounding_box[d, 1] = self.object_list[k].bounding_box[d, 1]

    # Gets the geometrical data that defines the deformable multi object, as a concatenated array.
    # We suppose that object data share the same last dimension (e.g. the list of list of points of vtks).
    def get_data(self):
        return np.concatenate([elt.get_data() for elt in self.object_list])

    def set_data(self, points):
        """
        points is a numpy array containing the new position of all the landmark points
        """
        assert len(points) == np.sum([elt.get_number_of_points() for elt in self.object_list]), "Number of points differ in template and data given to template"
        pos = 0
        for i,elt in enumerate(self.object_list):
            elt.set_points(points[pos:pos + elt.get_number_of_points()])
            pos += elt.get_number_of_points()

    def write(self, names):
        """
        Save the list of objects with the given names
        """
        assert len(names) == len(self.object_list), "Give as many names as objects to save multi-object"
        for i, name in enumerate(names):
            self.object_list[i].write(name)
