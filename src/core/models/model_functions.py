import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

import numpy as np
import math
import torch
from torch.autograd import Variable

from pydeformetrica.src.support.utilities.general_settings import Settings
from pydeformetrica.src.support.kernels.exact_kernel import ExactKernel


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


def compute_sobolev_gradient(template_gradient, smoothing_kernel_width, template, use_cholesky=False):
    """
    Smoothing of the template gradient (for landmarks).
    Fully torch input / outputs.
    """
    template_sobolev_gradient = Variable(torch.zeros(template_gradient.size()).type(Settings().tensor_scalar_type),
                                         requires_grad=False)

    kernel = ExactKernel()
    kernel.kernel_width = smoothing_kernel_width

    cursor = 0
    for template_object in template.object_list:
        # TODO : assert if obj is image or not.
        object_data = Variable(torch.from_numpy(
            template_object.get_points()).type(Settings().tensor_scalar_type), requires_grad=False)

        if use_cholesky:
            kernel_matrix_sqrt = torch.potrf(kernel.get_kernel_matrix(object_data))
            template_sobolev_gradient[cursor:cursor + len(object_data)] = torch.mm(
                kernel_matrix_sqrt, template_gradient[cursor:cursor + len(object_data)])
        else:
            template_sobolev_gradient[cursor:cursor + len(object_data)] = kernel.convolve(
                object_data, object_data, template_gradient[cursor:cursor + len(object_data)])

        cursor += len(object_data)

    return template_sobolev_gradient
