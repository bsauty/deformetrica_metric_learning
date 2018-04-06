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
        self.is_modified = True

        self.affine = None
        self.corner_points = None
        self.bounding_box = None

        self.intensities = None  # Numpy array.
        self.intensities_torch = None

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

    def set_affine(self, affine_matrix):
        """
        The affine matrix A is a 4x4 matrix that gives the correspondence between the voxel coordinates and their
        spatial positions in the 3D space: (x, y, z, 1) = A (u, v, w, 1).
        See the nibabel documentation for further details (the same attribute name is used here).
        """
        self.affine = affine_matrix

    def set_intensities(self, intensities):
        self.is_modified = True
        self.intensities = intensities

    def get_intensities(self):
        return self.intensities

    def get_intensities_torch(self):
        return self.intensities_torch

    def get_points(self):

        image_shape = self.intensities.shape
        dimension = Settings().dimension

        axes = []
        for d in range(dimension):
            axe = np.linspace(self.corner_points[0, d], self.corner_points[2 ** d, d], image_shape[d])
            axes.append(axe)

        points = np.array(np.meshgrid(*axes))
        for d in range(dimension):
            points = np.swapaxes(points, d, d + 1)

        return points

    def get_deformed_intensities(self, deformed_points, intensities):
        # Interpolation. Numpy function ?
        pass

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    # Update the relevant information.
    def update(self):
        if self.is_modified:
            self.update_corner_point_positions()
            self.update_bounding_box()
            self.intensities_torch = Variable(torch.from_numpy(self.intensities).type(Settings().tensor_scalar_type))
            self.is_modified = False

    def update_bounding_box(self):
        """
        Compute a tight bounding box that contains all the 2/3D-embedded image data.
        """
        dimension = Settings().dimension
        self.bounding_box = np.zeros((dimension, 2))
        for d in range(dimension):
            self.bounding_box[d, 0] = np.min(self.corner_points[:, d])
            self.bounding_box[d, 1] = np.max(self.corner_points[:, d])

    def write(self, name):
        raise RuntimeError("Writing not implemented for image yet.")

    ####################################################################################################################
    ### Utility methods:
    ####################################################################################################################

    def update_corner_point_positions(self):

        dimension = Settings().dimension
        if dimension == 2:
            corner_points = np.zeros((4, 2))
            umax, vmax = np.subtract(self.intensities.shape, (1, 1))
            corner_points[0] = np.dot(self.affine[0:2, 0:2], np.array([0, 0])) + self.affine[0:2, 2]
            corner_points[1] = np.dot(self.affine[0:2, 0:2], np.array([umax, 0])) + self.affine[0:2, 2]
            corner_points[2] = np.dot(self.affine[0:2, 0:2], np.array([0, vmax])) + self.affine[0:2, 2]
            corner_points[3] = np.dot(self.affine[0:2, 0:2], np.array([umax, vmax])) + self.affine[0:2, 2]

        elif dimension == 2:
            corner_points = np.zeros((8, 3))
            umax, vmax, wmax = np.subtract(self.intensities.shape, (1, 1, 1))
            corner_points[0] = np.dot(self.affine[0:3, 0:3], np.array([0, 0, 0])) + self.affine[0:3, 3]
            corner_points[1] = np.dot(self.affine[0:3, 0:3], np.array([umax, 0, 0])) + self.affine[0:3, 3]
            corner_points[2] = np.dot(self.affine[0:3, 0:3], np.array([0, vmax, 0])) + self.affine[0:3, 3]
            corner_points[3] = np.dot(self.affine[0:3, 0:3], np.array([umax, vmax, 0])) + self.affine[0:3, 3]
            corner_points[4] = np.dot(self.affine[0:3, 0:3], np.array([0, 0, wmax])) + self.affine[0:3, 3]
            corner_points[5] = np.dot(self.affine[0:3, 0:3], np.array([umax, 0, wmax])) + self.affine[0:3, 3]
            corner_points[6] = np.dot(self.affine[0:3, 0:3], np.array([0, vmax, wmax])) + self.affine[0:3, 3]
            corner_points[7] = np.dot(self.affine[0:3, 0:3], np.array([umax, vmax, wmax])) + self.affine[0:3, 3]

        else:
            raise RuntimeError('Unvalid dimension: %d' % dimension)

        self.corner_points = corner_points
