import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

import torch
import warnings
import time

from pydeformetrica.src.support.utilities.general_settings import Settings
from pydeformetrica.src.support.kernels.kernel_functions import create_kernel
from pydeformetrica.src.in_out.dataset_functions import create_template_metadata
from pydeformetrica.src.core.model_tools.deformations.geodesic import Geodesic
from pydeformetrica.src.core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from src.in_out.utils import *
from torch.autograd import Variable

def compute_parallel_transport(xml_parameters):
    """
    Takes as input an observation, a set of cp and mom which define the main geodesic, and another set of cp and mom describing the registration.
    Exp-parallel and geodesic-parallel are the two possible modes.
    """

    assert not xml_parameters.initial_control_points is None, "Please provide initial control points"
    assert not xml_parameters.initial_momenta is None, "Please provide initial momenta"
    assert not xml_parameters.initial_momenta_to_transport is None, "Please provide initial momenta to transport"

    control_points = read_2D_array(xml_parameters.initial_control_points)
    initial_momenta = read_2D_array(xml_parameters.initial_momenta)
    initial_momenta_to_transport = read_momenta(xml_parameters.initial_momenta_to_transport)[0]

    kernel = create_kernel('exact', xml_parameters.deformation_kernel_width)

    if xml_parameters.initial_control_points_to_transport is None:
        msg = "initial-control-points-to-transport was not specified, I amm assuming they are the same as initial-control-points"
        warnings.warn(msg)
        control_points_to_transport = control_points
        need_to_project_initial_momenta = False
    else:
        control_points_to_transport = read_2D_array(xml_parameters.initial_control_points_to_transport)
        need_to_project_initial_momenta = True

    control_points_torch = Variable(torch.from_numpy(control_points).type(Settings().tensor_scalar_type))
    initial_momenta_torch = Variable(torch.from_numpy(initial_momenta).type(Settings().tensor_scalar_type))
    initial_momenta_to_transport_torch = Variable(torch.from_numpy(initial_momenta_to_transport).type(Settings().tensor_scalar_type))

    #We start by projecting the initial momenta if they are not carried at the right control points.

    if need_to_project_initial_momenta:
        control_points_to_transport_torch = Variable(torch.from_numpy(control_points_to_transport).type(Settings().tensor_scalar_type))
        velocity = kernel.convolve(control_points_to_transport_torch, control_points_torch, initial_momenta_to_transport_torch)
        kernel_matrix = kernel.get_kernel_matrix(control_points_torch)
        cholesky_kernel_matrix = torch.potrf(kernel_matrix)
        projected_momenta = torch.potrs(velocity, cholesky_kernel_matrix).squeeze()

    else:
        projected_momenta = initial_momenta_to_transport_torch


    if xml_parameters.use_exp_parallelization in [None, True]:
        _exp_parallelize(control_points_torch, initial_momenta_torch, projected_momenta, xml_parameters)

    else:
        _geodesic_parallelize(control_points_torch, initial_momenta_torch, projected_momenta, xml_parameters)


def _exp_parallelize(control_points, initial_momenta, projected_momenta, xml_parameters):
    objects_list, objects_name, objects_name_extension, _, _ = create_template_metadata(xml_parameters.template_specifications)
    template = DeformableMultiObject()
    template.object_list = objects_list
    template.update()

    template_data = template.get_data()
    template_data_torch = Variable(torch.from_numpy(template_data).type(Settings().tensor_scalar_type))

    geodesic = Geodesic()
    geodesic.concentration_of_time_points = xml_parameters.number_of_time_points
    geodesic.set_kernel(create_kernel(xml_parameters.deformation_kernel_type, xml_parameters.deformation_kernel_width))

    #Those are mandatory parameters.
    assert geodesic.tmin != -float("inf"), "Please specify a minimum time for the geodesic trajectory"
    assert geodesic.tmax != float("inf"), "Please specify a maximum time for the geodesic trajectory"

    geodesic.tmin = xml_parameters.tmin
    geodesic.tmax = xml_parameters.tmax
    if xml_parameters.t0 is None:
        geodesic.t0 = geodesic.tmin
    else:
        geodesic.t0 = xml_parameters.t0


    geodesic.set_momenta_t0(initial_momenta)
    geodesic.set_control_points_t0(control_points)
    geodesic.set_template_data_t0(template_data_torch)
    geodesic.update()

    #We write the flow of the geodesic

    geodesic.write_flow("Regression", objects_name, objects_name_extension, template)

    #Now we transport!
    parallel_transport_trajectory = geodesic.parallel_transport(projected_momenta)

    #Getting trajectory caracteristics:
    times = geodesic.get_times()
    control_points_traj = geodesic.get_control_points_trajectory()
    momenta_traj = geodesic.get_momenta_trajectory()
    template_data_traj = geodesic.get_template_trajectory()

    other_geodesic = Geodesic()
    other_geodesic.number_of_time_points = xml_parameters.transported_trajectory_number_of_time_points
    other_geodesic.set_kernel(create_kernel(xml_parameters.deformation_kernel_type, xml_parameters.deformation_kernel_width))
    other_geodesic.tmin = xml_parameters.transported_trajectory_tmin
    other_geodesic.tmax = xml_parameters.transported_trajectory_tmax
    other_geodesic.t0 = other_geodesic.tmin


    #We save this trajectory, and the corresponding shape trajectory
    for i, (time, cp, mom, transported_mom, td) in enumerate(zip(times, control_points_traj, momenta_traj, parallel_transport_trajectory, template_data_traj)):
        #Writing the momenta/cps
        write_2D_array(cp.data.numpy(), "Control_Points_tp_" + str(i) + "__age_" + str(time) + ".txt")
        write_momenta(mom.data.numpy(), "Momenta_tp_" + str(i) + "__age_" + str(time) + ".txt")
        write_momenta(transported_mom.data.numpy(), "Transported_Momenta_tp_" + str(i) + "__age_" + str(time) + ".txt")

        #We make an assert here.
        momenta_deformetrica = read_2D_array("../alien_deformetrica/output/TransportedMomentas_" + str(i)+".txt")
        print("Error compared to deformetrica :", np.linalg.norm(transported_mom.data.numpy() - momenta_deformetrica)/np.linalg.norm(momenta_deformetrica))



        #Shooting from the geodesic:
        other_geodesic.set_template_data_t0(td)
        other_geodesic.set_control_points_t0(cp)
        other_geodesic.set_momenta_t0(transported_mom)
        other_geodesic.update()

        parallel_td = other_geodesic.get_template_data(other_geodesic.tmax)
        template.set_data(parallel_td)
        names = [objects_name[k] + "_parallel_curve_tp_"+ str(i) + "__age_" + str(time) + "_" + objects_name_extension[k] for k in range(len(objects_name))]
        template.write(names)



def _geodesic_parallelize(control_points, initial_momenta, projected_momenta, xml_parameters):
    print("Geodesic parallelization not implemented yet")
    pass




