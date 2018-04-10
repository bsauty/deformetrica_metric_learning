import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../')

import numpy as np
import torch
from torch.autograd import Variable

from numba import jit

import PIL.Image as pimg

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
        clone.is_modified = True

        clone.affine = np.copy(self.affine)
        clone.corner_points = np.copy(self.corner_points)
        clone.bounding_box = np.copy(self.bounding_box)

        clone.intensities = np.copy(self.intensities)
        clone.intensities_torch = self.intensities_torch.clone()
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
        for d in range(dimension - 1, -1, -1):
            axe = np.linspace(self.corner_points[0, d], self.corner_points[2 ** d, d], image_shape[d])
            axes.append(axe)

        points = np.array(np.meshgrid(*axes)[::-1])
        for d in range(dimension):
            points = np.swapaxes(points, d, d + 1)

        return points

    @jit(parallel=True)
    def get_deformed_intensities(self, deformed_points, intensities):
        """
        Torch input / output.
        Interpolation function with zero-padding.
        """
        dimension = Settings().dimension
        image_shape = self.intensities.shape
        deformed_voxels = self._compute_deformed_voxels(deformed_points)
        deformed_intensities = Variable(torch.zeros(intensities.size()).type(Settings().tensor_scalar_type))

        if dimension == 2:
            for u in range(image_shape[0]):
                for v in range(image_shape[1]):
                    deformed_pixel = deformed_voxels[u, v]
                    deformed_pixel_numpy = deformed_pixel.data.numpy()

                    # If the deformed pixel is outside of the original image, apply zero-padding.
                    # Could be optimized by not checking for the inner points ?
                    if deformed_pixel_numpy[0] < 0 or deformed_pixel_numpy[0] > image_shape[0] - 1 \
                            or deformed_pixel_numpy[1] < 0 or deformed_pixel_numpy[1] > image_shape[1] - 1:
                        continue

                    # Regular case.
                    iu1 = int(deformed_pixel[0])
                    iv1 = int(deformed_pixel[1])
                    iu2 = min(iu1 + 1, image_shape[0] - 1)
                    iv2 = min(iv1 + 1, image_shape[1] - 1)
                    fu = deformed_pixel[0] - iu1
                    fv = deformed_pixel[1] - iv1
                    gu = iu1 + 1 - deformed_pixel[0]
                    gv = iv1 + 1 - deformed_pixel[1]
                    deformed_intensities[u, v] = intensities[iu1, iv1] * gu * gv + \
                                                 intensities[iu1, iv2] * gu * fv + \
                                                 intensities[iu2, iv1] * fu * gv + \
                                                 intensities[iu2, iv2] * fu * fv

        elif dimension == 3:
            for u in range(image_shape[0]):
                for v in range(image_shape[1]):
                    for w in range(image_shape[2]):
                        deformed_voxel = deformed_voxels[u, v, w]
                        deformed_voxel_numpy = deformed_voxel.data.numpy()

                        # If the deformed voxel is outside of the original image, apply zero-padding.
                        # Could be optimized by not checking for the inner points ?
                        if deformed_voxel_numpy[0] <= 0 or deformed_voxel_numpy[0] >= image_shape[0] - 1 \
                                or deformed_voxel_numpy[1] <= 0 or deformed_voxel_numpy[1] >= image_shape[1] - 1 \
                                or deformed_voxel_numpy[2] <= 0 or deformed_voxel_numpy[2] >= image_shape[2] - 1:
                            continue

                        # Regular case.
                        iu1 = int(deformed_voxel[0])
                        iv1 = int(deformed_voxel[1])
                        iw1 = int(deformed_voxel[2])
                        iu2 = min(iu1 + 1, image_shape[0] - 1)
                        iv2 = min(iv1 + 1, image_shape[1] - 1)
                        iw2 = min(iw1 + 1, image_shape[2] - 1)
                        fu = deformed_voxel[0] - iu1
                        fv = deformed_voxel[1] - iv1
                        fw = deformed_voxel[2] - iw1
                        gu = iu1 + 1 - deformed_voxel[0]
                        gv = iv1 + 1 - deformed_voxel[1]
                        gw = iw1 + 1 - deformed_voxel[2]
                        deformed_intensities[u, v, w] = intensities[iu1, iv1, iw1] * gu * gv * gw + \
                                                        intensities[iu1, iv1, iw2] * gu * gv * fw + \
                                                        intensities[iu1, iv2, iw1] * gu * fv * gw + \
                                                        intensities[iu1, iv2, iw2] * gu * fv * fw + \
                                                        intensities[iu2, iv1, iw1] * fu * gv * gw + \
                                                        intensities[iu2, iv1, iw2] * fu * gv * fw + \
                                                        intensities[iu2, iv2, iw1] * fu * fv * gw + \
                                                        intensities[iu2, iv2, iw2] * fu * fv * fw

        else:
            raise RuntimeError('Incorrect dimension of the ambient space: %d' % dimension)

        return deformed_intensities

    def _compute_deformed_voxels(self, deformed_points):
        if (self.affine == np.eye(Settings().dimension + 1)).all():
            return deformed_points
        else:
            raise RuntimeError('_compute_deformed_voxels not implemented yet. Apply the inverse affine transform.')

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    # Update the relevant information.
    def update(self):
        if self.is_modified:
            self._update_corner_point_positions()
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

    def write(self, name, intensities=None):

        if intensities is None:
            intensities = self.get_intensities()

        if name.find(".png") > 0:
            pimg.fromarray((np.clip(intensities, 0, 1) * 255).astype('uint8')).save(
                os.path.join(Settings().output_dir, name))

        else:
            raise ValueError('Writing images with the given extension "%s" is not coded yet.' % name)

    ####################################################################################################################
    ### Utility methods:
    ####################################################################################################################

    def _update_corner_point_positions(self):

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
