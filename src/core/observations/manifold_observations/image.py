import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../')

import numpy as np
import torch
from torch.autograd import Variable

from pydeformetrica.src.support.utilities.general_settings import Settings


class Image:
    """
    Landmarks (i.e. labelled point sets).
    The Landmark class represents a set of labelled points. This class assumes that the source and the target
    have the same number of points with a point-to-point correspondence.

    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    # Constructor.
    def __init__(self):
        self.type = 'Image'
        self.intensities = None  # Numpy array.
        self.intensities_torch = None
        self.is_modified = True

    # Clone.
    def clone(self):
        clone = Image()
        clone.intensities = np.copy(self.intensities)
        clone.is_modified = True
        clone.update()
        return clone

    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################

    def get_number_of_points(self):
        raise RuntimeError("Not implemented for Image yet.")

    def set_points(self, points):
        """
        Sets the list of points of the poly data, to save at the end.
        """
        self.is_modified = True
        self.intensities = points

    # Gets the geometrical data that defines the landmark object, as a matrix list.
    def get_points(self):
        return self.intensities

    def get_points_torch(self):
        return self.intensities_torch

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    # Update the relevant information.
    def update(self):
        if self.is_modified:
            self.intensities_torch = Variable(torch.from_numpy(self.intensities).type(Settings().tensor_scalar_type))
            self.is_modified = False

    def write(self, name):
        raise RuntimeError("Writing not implemented for image yet.")