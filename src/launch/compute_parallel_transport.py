import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

import torch
import warnings
import time

from pydeformetrica.src.support.utilities.general_settings import Settings
from pydeformetrica.src.support.kernels.kernel_functions import create_kernel
from pydeformetrica.src.in_out.dataset_functions import create_template_metada
from pydeformetrica.src.core.model_tools.deformations.exponential import Exponential
from pydeformetrica.src.core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject


def compute_parallel_transport(xml_parameters):
    """
    Takes as input an observation, a set of cp and mom which define the main geodesic, and another set of cp and mom describing the registration.
    Exp-parallel and geodesic-parallel are the two possible modes.
    """

    assert not xml_parameters.initial_control_points is None, "Please provide initial control points"
    assert not xml_parameters.initial_momenta is None, "Please provide initial momenta"
    assert not xml_parameters.initial_momenta_to_transport is None, "Please provide initial momenta to transport"

    control_points = read_2D_array(xml_parameters.initial_control_points)
    initial_momenta = read_momenta(xml_parameters.initial_momenta)
    initial_momenta_to_transport = read_momenta(xml_parameters.initial_momenta)

    kernel = create_kernel('exact', xml_parameters.deformation_kernel_width)

    if xml_parameters.initial_control_points_to_transport is None:
        msg = "initial-control-points-to-transport was not specified, I amm assuming they are the same as initial-control-points"
        warnings.warn(msg)
        control_points_to_transport = control_points
        need_to_project_initial_momenta = False
    else:
        control_points = read_2D_array(xml_parameters.initial_control_points_to_transport)
        need_to_project_initial_momenta = True

    #We start by projecting the initial momenta if they are not carried at the right control points.

    if need_to_project_initial_momenta:
        velocity = kernel.convolve(control_points_to_transport, control_points, initial_momenta_to_transport)
        kernel_matrix = kernel.get_kernel_matrix(control_points)
        cholesky_kernel_matrix = torch.potrf(kernel_matrix)
        projected_momenta = torch.potrs(velocity.unsqueeze(1), cholesky_kernel_matrix).squeeze()


    else:
        projected_momenta = initial_momenta_to_transport


    if self.use_exp_parallelization:
        _exp_parallelize(control_points, initial_momenta, projected_momenta, xml_parameters)

    else:
        _geodesic_parallelize(control_points, initial_momenta, projected_momenta, xml_parameters)


def _exp_parallelize(control_points, initial_momenta, projected_momenta, xml_parameters):
    objects_list, objects_name, objects_name_extension, _, _, _ = create_template_metada()
    template = DeformableMultiObject()
    template.object_list = objects_list
    template.update()

    template_data = template.get_data()

    diffeo = Exponential()
    diffeo.number_of_time_points =
    diffeo.kernel = create_kernel(xml_parameters.deformation_kernel_type, xml_parameters.deformation_kernel_width)
    diffeo.set_initial_momenta_from_numpy(initial_momenta)
    diffeo.set_initial_control_points_from_numpy(control_points)
    diffeo.set_template_data_from_numpy(template_data)
    diffeo.update()

    #Now we transport!





def _geodesic_parallelize(control_points, initial_momenta, projected_momenta, xml_parameters):
    pass




