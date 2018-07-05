import math

import numpy as np
import scipy
import torch
from torch.autograd import Variable

import support.kernels as kernel_factory
from support.utilities.general_settings import Settings
from in_out.image_functions import points_to_voxels_transform, metric_to_image_radial_length


def create_regular_grid_of_points(box, spacing):
    """
    Creates a regular grid of 2D or 3D points, as a numpy array of size nb_of_points x dimension.
    box: (dimension, 2)
    """

    dimension = Settings().dimension

    axis = []
    for d in range(dimension):
        min = box[d, 0]
        max = box[d, 1]
        length = max - min
        assert (length > 0)

        offset = 0.5 * (length - spacing * math.floor(length / spacing))
        axis.append(np.arange(min + offset, max + 1e-10, spacing))

    if dimension == 1:
        control_points = np.zeros((len(axis[0]), dimension))
        control_points[:, 0] = axis[0].flatten()

    elif dimension == 2:
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

    elif dimension == 4:
        x_axis, y_axis, z_axis, t_axis = np.meshgrid(axis[0], axis[1], axis[2], axis[3])

        assert (x_axis.shape == y_axis.shape)
        assert (x_axis.shape == z_axis.shape)
        number_of_control_points = x_axis.flatten().shape[0]
        control_points = np.zeros((number_of_control_points, dimension))

        control_points[:, 0] = x_axis.flatten()
        control_points[:, 1] = y_axis.flatten()
        control_points[:, 2] = z_axis.flatten()
        control_points[:, 3] = t_axis.flatten()

    else:
        raise RuntimeError('Invalid ambient space dimension.')

    return control_points


def remove_useless_control_points(control_points, image, kernel_width):
    control_voxels = points_to_voxels_transform(control_points, image.affine)  # To be modified if image + mesh case.
    kernel_voxel_width = metric_to_image_radial_length(kernel_width, image.affine)

    dimension = Settings().dimension
    intensities = image.get_intensities()
    image_shape = intensities.shape

    threshold = 1e-5
    region_size = 2 * kernel_voxel_width

    final_control_points = []
    for control_point, control_voxel in zip(control_points, control_voxels):

        axes = []
        for d in range(dimension):
            axe = np.arange(max(int(control_voxel[d] - region_size), 0),
                            min(int(control_voxel[d] + region_size), image_shape[d] - 1))
            axes.append(axe)

        neighbouring_voxels = np.array(np.meshgrid(*axes))
        for d in range(dimension):
            neighbouring_voxels = np.swapaxes(neighbouring_voxels, d, d + 1)
        neighbouring_voxels = neighbouring_voxels.reshape(-1, dimension)

        if (dimension == 2 and np.any(intensities[neighbouring_voxels[:, 0],
                                                  neighbouring_voxels[:, 1]] > threshold)) \
                or (dimension == 3 and np.any(intensities[neighbouring_voxels[:, 0],
                                                          neighbouring_voxels[:, 1],
                                                          neighbouring_voxels[:, 2]] > threshold)):
            final_control_points.append(control_point)

    return np.array(final_control_points)


def compute_sobolev_gradient(template_gradient, smoothing_kernel_width, template, square_root=False):
    """
    Smoothing of the template gradient (for landmarks).
    Fully torch input / outputs.
    """
    template_sobolev_gradient = torch.zeros(template_gradient.size()).type(Settings().tensor_scalar_type)
    kernel = kernel_factory.factory(kernel_factory.Type.TorchKernel, smoothing_kernel_width)

    cursor = 0
    for template_object in template.object_list:
        # TODO : assert if obj is image or not.
        object_data = torch.from_numpy(template_object.get_points()).type(Settings().tensor_scalar_type)

        if square_root:
            kernel_matrix = kernel.get_kernel_matrix(object_data).data.numpy()
            kernel_matrix_sqrt = Variable(torch.from_numpy(
                scipy.linalg.sqrtm(kernel_matrix).real).type(Settings().tensor_scalar_type), requires_grad=False)
            template_sobolev_gradient[cursor:cursor + len(object_data)] = torch.mm(
                kernel_matrix_sqrt, template_gradient[cursor:cursor + len(object_data)])
        else:
            template_sobolev_gradient[cursor:cursor + len(object_data)] = kernel.convolve(
                object_data, object_data, template_gradient[cursor:cursor + len(object_data)])

        cursor += len(object_data)

    return template_sobolev_gradient
