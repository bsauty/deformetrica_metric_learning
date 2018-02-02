import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

import torch
from torch.autograd import Variable
import warnings
import time

from pydeformetrica.src.core.models.deterministic_atlas import DeterministicAtlas
from pydeformetrica.src.core.estimators.scipy_optimize import ScipyOptimize
from pydeformetrica.src.core.estimators.gradient_ascent import GradientAscent
from pydeformetrica.src.support.utilities.general_settings import Settings
from pydeformetrica.src.support.kernels.kernel_functions import create_kernel
from pydeformetrica.src.in_out.dataset_functions import create_dataset
from src.in_out.utils import *


def instantiate_deterministic_atlas_model(xml_parameters, dataset=None, ignore_noise_variance=False):
    model = DeterministicAtlas()
    model.number_of_subjects = dataset.number_of_subjects

    # Deformation object -----------------------------------------------------------------------------------------------
    model.exponential.kernel = create_kernel(xml_parameters.deformation_kernel_type,
                                             xml_parameters.deformation_kernel_width)
    model.exponential.number_of_time_points = xml_parameters.number_of_time_points
    model.exponential.set_use_rk2(xml_parameters.use_rk2)

    # Initial fixed effects --------------------------------------------------------------------------------------------
    # Template.
    model.freeze_template = xml_parameters.freeze_template  # this should happen before the init of the template and the cps
    model.initialize_template_attributes(xml_parameters.template_specifications)
    model.use_sobolev_gradient = xml_parameters.use_sobolev_gradient
    model.smoothing_kernel_width = xml_parameters.deformation_kernel_width * xml_parameters.sobolev_kernel_width_ratio

    # Control points.
    model.freeze_control_points = xml_parameters.freeze_control_points
    if xml_parameters.initial_control_points is not None:
        control_points = read_2D_array(xml_parameters.initial_control_points)
        print(">> Reading " + str(len(control_points)) + " initial control points from file "
              + xml_parameters.initial_control_points)
        model.set_control_points(control_points)
    else:
        model.initial_cp_spacing = xml_parameters.initial_cp_spacing

    if not xml_parameters.control_points_on_shape is None:
        model.control_points_on_shape = xml_parameters.control_points_on_shape

    # Momenta.
    if xml_parameters.initial_momenta is not None:
        momenta = read_momenta(xml_parameters.initial_momenta)
        print('>> Reading initial momenta from file: ' + xml_parameters.initial_momenta)
        model.set_momenta(momenta)

    # Final initialization steps by the model object itself ------------------------------------------------------------
    model.update()

    # Special case of the noise variance hyperparameter ----------------------------------------------------------------
    # Compute residuals if needed.
    if not ignore_noise_variance and np.min(model.objects_noise_variance) < 0:

        template_data_torch = Variable(torch.from_numpy(
            model.get_template_data()).type(Settings().tensor_scalar_type), requires_grad=False)
        control_points_torch = Variable(torch.from_numpy(
            model.get_control_points()).type(Settings().tensor_scalar_type), requires_grad=False)
        momenta_torch = Variable(torch.from_numpy(
            model.get_momenta()).type(Settings().tensor_scalar_type), requires_grad=False)

        targets = dataset.deformable_objects
        targets = [target[0] for target in targets]
        residuals_torch = []
        model.exponential.set_initial_template_data(template_data_torch)
        model.exponential.set_initial_control_points(control_points_torch)
        for i, target in enumerate(targets):
            model.exponential.set_initial_momenta(momenta_torch[i])
            model.exponential.update()
            deformed_points = model.exponential.get_template_data()
            residuals_torch.append(model.multi_object_attachment.compute_distances(
                deformed_points, model.template, target))

        residuals = np.zeros((model.number_of_objects,))
        for i in range(len(residuals_torch)):
            residuals += residuals_torch[i].data.numpy()

        # Initialize the noise variance hyperparameter.
        for k, obj in enumerate(xml_parameters.template_specifications.values()):
            if model.objects_noise_variance[k] < 0:
                model.objects_noise_variance[k] = 0.05 * residuals[k] / float(model.number_of_subjects)

    # Return the initialized model.
    return model


def estimate_deterministic_atlas(xml_parameters):

    print('[ estimate_deterministic_atlas function ]')
    print('')

    """
    Create the dataset object.
    """

    dataset = create_dataset(xml_parameters.dataset_filenames, xml_parameters.visit_ages,
                             xml_parameters.subject_ids, xml_parameters.template_specifications)

    assert (dataset.is_cross_sectional()), "Cannot estimate an atlas from a non-cross-sectional dataset."

    """
    Create the model object.
    """

    model = instantiate_deterministic_atlas_model(xml_parameters, dataset)

    """
    Create the estimator object.
    """

    if xml_parameters.optimization_method_type == 'GradientAscent'.lower():
        estimator = GradientAscent()
        estimator.initial_step_size = xml_parameters.initial_step_size
        estimator.line_search_shrink = xml_parameters.line_search_shrink
        estimator.line_search_expand = xml_parameters.line_search_expand

    elif xml_parameters.optimization_method_type == 'ScipyLBFGS'.lower():
        estimator = ScipyOptimize()
        estimator.memory_length = xml_parameters.memory_length
        if not model.freeze_template and model.use_sobolev_gradient and estimator.memory_length > 1:
            estimator.memory_length = 1
            msg = 'Impossible to use a Sobolev gradient for the template data with the ScipyLBFGS estimator memory ' \
                  'length being larger than 1. Overriding the "memory_length" option, now set to "1".'
            warnings.warn(msg)

    else:
        estimator = GradientAscent()
        estimator.initial_step_size = xml_parameters.initial_step_size
        estimator.max_line_search_iterations = xml_parameters.max_line_search_iterations
        estimator.line_search_shrink = xml_parameters.line_search_shrink
        estimator.line_search_expand = xml_parameters.line_search_expand

        msg = 'Unknown optimization-method-type: \"' + xml_parameters.optimization_method_type \
              + '\". Defaulting to GradientAscent.'
        warnings.warn(msg)

    estimator.max_iterations = xml_parameters.max_iterations
    estimator.max_line_search_iterations = xml_parameters.max_line_search_iterations
    estimator.convergence_tolerance = xml_parameters.convergence_tolerance

    estimator.print_every_n_iters = xml_parameters.print_every_n_iters
    estimator.save_every_n_iters = xml_parameters.save_every_n_iters

    estimator.dataset = dataset
    estimator.statistical_model = model

    """
    Launch.
    """

    if not os.path.exists(Settings().output_dir):
        os.makedirs(Settings().output_dir)

    model.name = 'DeterministicAtlas'

    start_time = time.time()
    estimator.update()
    estimator.write()
    end_time = time.time()
    print('>> Estimation took: ' + str(time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))))

    # Can do extra stuff!