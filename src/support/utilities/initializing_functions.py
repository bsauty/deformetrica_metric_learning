import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

import numpy as np
import math

from pydeformetrica.src.support.utilities.general_settings import Settings


def create_regular_grid_of_points(box, spacing):
    """
    Creates a regular grid of 2D or 3D points, as a numpy array of size nb_of_points x dimension.
    """

    dimension = Settings().dimension

    axis = []
    for d in range(dimension):
        min = box[d, 0]
        max = box[d, 1]
        length = max - min
        assert (length > 0)

        offset = 0.5 * (length - spacing * math.floor(length / spacing))
        axis.append(np.arange(min + offset, max, spacing))

    if dimension == 2:
        x_axis, y_axis = np.meshgrid(axis[0], axis[1])

        assert (x_axis.shape == y_axis.shape)
        number_of_control_points = x_axis.flatten().shape[0]
        control_points = np.zeros((number_of_control_points, dimension))

        control_points[:, 0] = x_axis.flatten()
        control_points[:, 1] = y_axis.flatten()

    elif dimension == 3:
        x_axis, y_axis, z_axis = np.meshgrid(axis[0], axis[1], axis[2])

        assert (x_axis.shape == y_axis.shape)
        assert (x_axis.shape == z_axis.shape)
        number_of_control_points = x_axis.flatten().shape[0]
        control_points = np.zeros((number_of_control_points, dimension))

        control_points[:, 0] = x_axis.flatten()
        control_points[:, 1] = y_axis.flatten()
        control_points[:, 2] = z_axis.flatten()

    else:
        raise RuntimeError('Invalid ambient space dimension.')

    return control_points
