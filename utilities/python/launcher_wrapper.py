import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

import numpy as np

from pydeformetrica.src.launch.estimate_deterministic_atlas import estimate_deterministic_atlas
from pydeformetrica.src.launch.estimate_geodesic_regression import estimate_geodesic_regression
from pydeformetrica.src.launch.compute_parallel_transport import compute_parallel_transport
from pydeformetrica.src.in_out. xml_parameters import XmlParameters
from pydeformetrica.src.support.utilities.general_settings import Settings
from pydeformetrica.src.in_out.array_readers_and_writers import *

#Functions used to script deformetrica. WORK IN PROGRESS, lots of parameters are hardcoded, cuda not managed...

def perform_registration(source_vtk, target_vtk, object_type, attachment_type, noise_std, object_id,
                         deformation_kernel_width, output_dir, attachment_kernel_width, subject_id='0',
                         deformation_kernel_type='Exact', attachment_kernel_type='Exact', freeze_cp=True, number_of_time_points=10,
                         control_points_on_shape=None, initial_step_size=1e-2):
    """
    Performs a registration, using the given parameters. It wraps estimate deterministic_atlas.
    lots of default parameters: optimization_method_type, initial_cp_spacing, use_rk2
    #ACHTUNG CUDA not managed here
    """
    xml_parameters = XmlParameters()

    # Model parameters
    xml_parameters.freeze_template = True
    xml_parameters.freeze_cp = True
    xml_parameters.initial_cp_spacing = deformation_kernel_width

    xml_parameters.use_cuda = True

    xml_parameters.control_points_on_shape = control_points_on_shape

    # Deformation parameters
    xml_parameters.deformation_kernel_width = deformation_kernel_width
    xml_parameters.deformation_kernel_type = deformation_kernel_type
    xml_parameters.number_of_time_points = number_of_time_points

    # Optimization parameters
    xml_parameters.use_rk2 = True
    # xml_parameters.optimization_method_type = 'ScipyLBFGS'.lower()
    xml_parameters.optimization_method_type = 'ScipyLBFGS'.lower()
    xml_parameters.initial_step_size = initial_step_size
    xml_parameters.max_iterations = 200
    xml_parameters.save_every_n_iters = 20
    xml_parameters.convergence_tolerance = 1e-5

    Settings().set_output_dir(output_dir)

    # Deformable objects parameters
    target = {object_id : target_vtk}
    xml_parameters.dataset_filenames = [[target]]
    xml_parameters.subject_ids = [subject_id]

    template_specifications = {}
    template_object = {'deformable_object_type': object_type.lower(),
                       'attachment_type': attachment_type.lower(),
                       'kernel_width' : attachment_kernel_width,
                       'kernel_type': attachment_kernel_type.lower(),
                        'noise_std' : noise_std,
                       'filename': source_vtk}

    template_specifications[object_id] = template_object

    xml_parameters.template_specifications = template_specifications

    xml_parameters._further_initialization()

    estimate_deterministic_atlas(xml_parameters)

    control_points = os.path.join(output_dir, "DeterministicAtlas__control_points.txt")
    momenta = os.path.join(output_dir, "DeterministicAtlas__momenta.txt")

    return control_points, momenta


def parallel_transport(template_vtk, object_type, object_id, deformation_kernel_width,
                       control_points, initial_momenta, initial_momenta_to_transport,
                       output_dir, initial_control_points_to_transport=None):
    xml_parameters = XmlParameters()

    xml_parameters.deformation_kernel_width = deformation_kernel_width
    xml_parameters.initial_cp_spacing = deformation_kernel_width
    xml_parameters.deformation_kernel_type = 'Exact'
    xml_parameters.number_of_time_points = 20
    xml_parameters.concentration_of_time_points = 200
    # xml_parameters.number_of_time_points = 50
    # xml_parameters.concentration_of_time_points = 50

    xml_parameters.tmin = 0.
    xml_parameters.tmax = 1.

    xml_parameters.use_rk2 = True

    xml_parameters.transported_trajectory_tmin = 0
    xml_parameters.transport_trajectory_t0 = 0
    xml_parameters.transported_trajectory_tmax = 1.

    xml_parameters.initial_control_points = control_points
    xml_parameters.initial_momenta = initial_momenta
    xml_parameters.initial_momenta_to_transport = initial_momenta_to_transport
    xml_parameters.initial_control_points_to_transport = initial_control_points_to_transport
    template_specifications = {}
    template_object = {'deformable_object_type': object_type.lower(),
                       'attachment_type': 'Landmark'.lower(),
                       'kernel_width': 'not_needed',
                       'kernel_type': 'not_needed',
                       'noise_std': 1.,
                       'filename': template_vtk}

    template_specifications[object_id] = template_object
    Settings().set_output_dir(output_dir)
    if not(os.path.isdir(output_dir)):
        os.mkdir(output_dir)

    xml_parameters.template_specifications = template_specifications

    xml_parameters._further_initialization()

    compute_parallel_transport(xml_parameters)

    # We now extract the final file path of the parallel curve (not very generic, for personal use...)
    return os.path.join(output_dir, object_id + "_parallel_curve_tp_"+
                        str(xml_parameters.concentration_of_time_points)+"__age_"+"1.0_.vtk")








