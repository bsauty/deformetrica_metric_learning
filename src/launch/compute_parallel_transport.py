import warnings

import torch
from torch.autograd import Variable

from core.model_tools.deformations.exponential import Exponential
from core.model_tools.deformations.geodesic import Geodesic
from core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from in_out.array_readers_and_writers import *
from in_out.dataset_functions import create_template_metadata
import support.kernel as kernel_factory


def compute_parallel_transport(xml_parameters):
    """
    Takes as input an observation, a set of cp and mom which define the main geodesic, and another set of cp and mom describing the registration.
    Exp-parallel and geodesic-parallel are the two possible modes.
    """

    assert not xml_parameters.initial_control_points is None, "Please provide initial control points"
    assert not xml_parameters.initial_momenta is None, "Please provide initial momenta"
    assert not xml_parameters.initial_momenta_to_transport is None, "Please provide initial momenta to transport"

    control_points = read_2D_array(xml_parameters.initial_control_points)
    initial_momenta = read_3D_array(xml_parameters.initial_momenta)
    initial_momenta_to_transport = read_3D_array(xml_parameters.initial_momenta_to_transport)

    kernel = kernel_factory.factory(kernel_factory.Type.ExactKernel, xml_parameters.deformation_kernel_width)

    if xml_parameters.initial_control_points_to_transport is None:
        msg = "initial-control-points-to-transport was not specified, I am assuming they are the same as initial-control-points"
        warnings.warn(msg)
        control_points_to_transport = control_points
        need_to_project_initial_momenta = False
    else:
        control_points_to_transport = read_2D_array(xml_parameters.initial_control_points_to_transport)
        need_to_project_initial_momenta = True

    control_points_torch = Variable(torch.from_numpy(control_points).type(Settings().tensor_scalar_type))
    initial_momenta_torch = Variable(torch.from_numpy(initial_momenta).type(Settings().tensor_scalar_type))
    initial_momenta_to_transport_torch = Variable(
        torch.from_numpy(initial_momenta_to_transport).type(Settings().tensor_scalar_type))

    # We start by projecting the initial momenta if they are not carried at the right control points.

    if need_to_project_initial_momenta:
        control_points_to_transport_torch = Variable(
            torch.from_numpy(control_points_to_transport).type(Settings().tensor_scalar_type))
        velocity = kernel.convolve(control_points_torch, control_points_to_transport_torch,
                                   initial_momenta_to_transport_torch)
        kernel_matrix = kernel.get_kernel_matrix(control_points_torch)
        cholesky_kernel_matrix = torch.potrf(kernel_matrix)
        # cholesky_kernel_matrix = Variable(torch.Tensor(np.linalg.cholesky(kernel_matrix.data.numpy())).type_as(kernel_matrix))#Dirty fix if pytorch fails.
        projected_momenta = torch.potrs(velocity, cholesky_kernel_matrix).squeeze().contiguous()

    else:
        projected_momenta = initial_momenta_to_transport_torch

    if xml_parameters.use_exp_parallelization in [None, True]:
        _exp_parallelize(control_points_torch, initial_momenta_torch, projected_momenta, xml_parameters)

    else:
        _geodesic_parallelize(control_points_torch, initial_momenta_torch, projected_momenta, xml_parameters)


def _exp_parallelize(control_points, initial_momenta, projected_momenta, xml_parameters):
    objects_list, objects_name, objects_name_extension, _, _ = create_template_metadata(
        xml_parameters.template_specifications)
    template = DeformableMultiObject()
    template.object_list = objects_list
    template.update()

    template_data = template.get_points()
    template_data_torch = Variable(torch.from_numpy(template_data).type(Settings().tensor_scalar_type))

    geodesic = Geodesic()
    geodesic.concentration_of_time_points = xml_parameters.concentration_of_time_points
    geodesic.set_kernel(kernel_factory.factory(xml_parameters.deformation_kernel_type, xml_parameters.deformation_kernel_width))
    geodesic.set_use_rk2(xml_parameters.use_rk2)

    # Those are mandatory parameters.
    assert xml_parameters.tmin != -float("inf"), "Please specify a minimum time for the geodesic trajectory"
    assert xml_parameters.tmax != float("inf"), "Please specify a maximum time for the geodesic trajectory"

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

    # We write the flow of the geodesic

    geodesic.write("Regression", objects_name, objects_name_extension, template)

    # Now we transport!
    parallel_transport_trajectory = geodesic.parallel_transport(projected_momenta)

    # Getting trajectory caracteristics:
    times = geodesic._get_times()
    control_points_traj = geodesic._get_control_points_trajectory()
    momenta_traj = geodesic._get_momenta_trajectory()
    template_data_traj = geodesic._get_template_data_trajectory()

    exponential = Exponential()
    exponential.number_of_time_points = xml_parameters.number_of_time_points
    exponential.set_kernel(
        kernel_factory.factory(xml_parameters.deformation_kernel_type, xml_parameters.deformation_kernel_width))
    exponential.set_use_rk2(xml_parameters.use_rk2)

    # We save this trajectory, and the corresponding shape trajectory
    for i, (time, cp, mom, transported_mom, td) in enumerate(
            zip(times, control_points_traj, momenta_traj, parallel_transport_trajectory, template_data_traj)):
        # Writing the momenta/cps
        write_2D_array(cp.data.numpy(), "control_Points_tp_" + str(i) + "__age_" + str(time) + ".txt")
        write_3D_array(mom.data.numpy(), "momenta_tp_" + str(i) + "__age_" + str(time) + ".txt")
        write_3D_array(transported_mom.data.numpy(), "transported_momenta_tp_" + str(i) + "__age_" + str(time) + ".txt")

        # Shooting from the geodesic:
        exponential.set_initial_template_data(td)
        exponential.set_initial_control_points(cp)
        exponential.set_initial_momenta(transported_mom)
        exponential.update()

        # Uncomment for massive writing, useful for debugging.
        # dir = "exp_"+str(i)+"_"+str(time)
        # if not(os.path.isdir(os.path.join(Settings().output_dir, dir))):
        #     os.mkdir(os.path.join(Settings().output_dir, dir))
        # exponential.write_flow([os.path.join(dir, elt) for elt in objects_name],
        #                        objects_name_extension,
        #                        template)
        # exponential.write_control_points_and_momenta_flow(os.path.join(dir, "cp_and_mom"))

        parallel_td = exponential.get_template_data()
        template.set_data(parallel_td)
        names = [
            objects_name[k] + "_parallel_curve_tp_" + str(i) + "__age_" + str(time) + "_" + objects_name_extension[k]
            for k in range(len(objects_name))]
        template.write(names)


def _geodesic_parallelize(control_points, initial_momenta, projected_momenta, xml_parameters):
    print("Geodesic parallelization not implemented yet")
    pass
