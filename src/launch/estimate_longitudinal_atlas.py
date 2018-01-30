import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

import torch
from torch.autograd import Variable
import warnings
import time

from pydeformetrica.src.core.models.longitudinal_atlas import LongitudinalAtlas
from pydeformetrica.src.core.estimators.scipy_optimize import ScipyOptimize
from pydeformetrica.src.core.estimators.gradient_ascent import GradientAscent
from pydeformetrica.src.core.estimators.mcmc_saem import McmcSaem
from pydeformetrica.src.core.estimator_tools.samplers.srw_mhwg_sampler import SrwMhwgSampler
from pydeformetrica.src.support.utilities.general_settings import Settings
from pydeformetrica.src.support.kernels.kernel_functions import create_kernel
from pydeformetrica.src.support.probability_distributions.multi_scalar_normal_distribution import \
    MultiScalarNormalDistribution
from pydeformetrica.src.in_out.dataset_functions import create_dataset
from src.in_out.utils import *


def estimate_longitudinal_atlas(xml_parameters):
    print('')
    print('[ estimate_longitudinal_atlas function ]')
    print('')

    """
    Create the dataset object.
    """

    dataset = create_dataset(xml_parameters.dataset_filenames, xml_parameters.visit_ages,
                             xml_parameters.subject_ids, xml_parameters.template_specifications)

    """
    Create the model object.
    """

    model = LongitudinalAtlas()

    # Deformation object -----------------------------------------------------------------------------------------------
    model.spatiotemporal_reference_frame.set_kernel(create_kernel(xml_parameters.deformation_kernel_type,
                                                                  xml_parameters.deformation_kernel_width))
    model.spatiotemporal_reference_frame.set_concentration_of_time_points(xml_parameters.concentration_of_time_points)
    model.spatiotemporal_reference_frame.set_number_of_time_points(xml_parameters.number_of_time_points)
    model.spatiotemporal_reference_frame.set_use_rk2(xml_parameters.use_rk2)

    # Initial fixed effects and associated priors ----------------------------------------------------------------------
    # Template.
    model.is_frozen['template_data'] = xml_parameters.freeze_template
    model.initialize_template_attributes(xml_parameters.template_specifications)
    model.use_sobolev_gradient = xml_parameters.use_sobolev_gradient
    model.smoothing_kernel_width = xml_parameters.deformation_kernel_width * xml_parameters.sobolev_kernel_width_ratio
    model.initialize_template_data_variables()

    # Control points.
    model.is_frozen['control_points'] = xml_parameters.freeze_control_points
    if xml_parameters.initial_control_points is not None:
        control_points = read_2D_array(xml_parameters.initial_control_points)
        print('>> Reading ' + str(len(control_points)) + ' initial control points from file: '
              + xml_parameters.initial_control_points)
        model.set_control_points(control_points)
    else:
        model.initial_cp_spacing = xml_parameters.initial_cp_spacing
    model.initialize_control_points_variables()

    # Momenta.
    model.is_frozen['momenta'] = xml_parameters.freeze_momenta
    if not xml_parameters.initial_momenta is None:
        momenta = read_momenta(xml_parameters.initial_momenta)
        print('>> Reading initial momenta from file: ' + xml_parameters.initial_control_points)
        model.set_momenta(momenta)
    model.initialize_momenta_variables()

    # Modulation matrix.
    model.is_frozen['modulation_matrix'] = xml_parameters.freeze_modulation_matrix
    if not xml_parameters.initial_modulation_matrix is None:
        modulation_matrix = read_2D_array(xml_parameters.initial_modulation_matrix)
        print('>> Reading ' + str(modulation_matrix.shape[1]) + '-source initial modulation matrix from file: '
              + xml_parameters.initial_modulation_matrix)
        model.set_modulation_matrix(modulation_matrix)
    else:
        model.number_of_sources = xml_parameters.number_of_sources
    model.initialize_modulation_matrix_variables()

    # Reference time.
    model.is_frozen['reference_time'] = xml_parameters.freeze_reference_time
    model.set_reference_time(xml_parameters.t0)
    model.priors['reference_time'].set_variance(xml_parameters.initial_time_shift_variance)
    model.initialize_reference_time_variables()

    # Time-shift variance.
    model.is_frozen['time_shift_variance'] = xml_parameters.freeze_time_shift_variance
    model.set_time_shift_variance(xml_parameters.initial_time_shift_variance)

    # Log-acceleration variance.
    model.is_frozen['log_acceleration_variance'] = xml_parameters.freeze_log_acceleration_variance
    model.set_log_acceleration_variance(xml_parameters.initial_log_acceleration_variance)

    # Initial random effects realizations ------------------------------------------------------------------------------
    sources = np.zeros((dataset.number_of_subjects, model.number_of_sources))
    onset_ages = np.zeros((dataset.number_of_subjects,)) + model.get_reference_time()
    log_accelerations = np.zeros((dataset.number_of_subjects,))

    # Special case of the noise variance -------------------------------------------------------------------------------
    model.is_frozen['noise_variance'] = xml_parameters.freeze_noise_variance
    initial_noise_variance = model.get_noise_variance()

    # Compute residuals if needed.
    if (np.min(initial_noise_variance) < 0) or (not model.is_frozen['noise_variance']):

        template_data_torch = Variable(torch.from_numpy(
            model.get_template_data()).type(Settings().tensor_scalar_type), requires_grad=False)
        control_points_torch = Variable(torch.from_numpy(
            model.get_control_points()).type(Settings().tensor_scalar_type), requires_grad=False)
        momenta_torch = Variable(torch.from_numpy(
            model.get_momenta()).type(Settings().tensor_scalar_type), requires_grad=False)
        modulation_matrix_torch = Variable(torch.from_numpy(
            model.get_modulation_matrix()).type(Settings().tensor_scalar_type), requires_grad=False)
        sources_torch = Variable(torch.from_numpy(sources).type(Settings().tensor_scalar_type), requires_grad=False)
        onset_ages_torch = Variable(torch.from_numpy(onset_ages).type(Settings().tensor_scalar_type),
                                    requires_grad=False)
        log_accelerations_torch = Variable(torch.from_numpy(
            log_accelerations).type(Settings().tensor_scalar_type), requires_grad=False)
        residuals_torch = model._compute_residuals(
            dataset, template_data_torch, control_points_torch, momenta_torch, modulation_matrix_torch,
            sources_torch, onset_ages_torch, log_accelerations_torch)
        residuals = np.zeros((model.number_of_objects,))
        for i in range(len(residuals_torch)):
            for j in range(len(residuals_torch[i])):
                residuals += residuals_torch[i][j].data.numpy()

    # Initialize noise variance fixed effect, and the noise variance prior if needed.
    for k, obj in enumerate(xml_parameters.template_specifications.values()):
        dof = dataset.total_number_of_observations * obj['noise_variance_prior_normalized_dof'] * \
              model.objects_noise_dimension[k]
        nv = 0.01 * residuals[k] / dof
        if initial_noise_variance[k] < 0: initial_noise_variance[k] = nv

        if not model.is_frozen['noise_variance']:
            model.priors['noise_variance'].degrees_of_freedom.append(dof)
            if obj['noise_variance_prior_scale_std'] is None:
                model.priors['noise_variance'].scale_scalars.append(nv)
            else:
                model.priors['noise_variance'].scale_scalars.append(obj['noise_variance_prior_scale_std'] ** 2)

    # Final initialization steps by the model object itself ------------------------------------------------------------
    model.update()

    """
    Create the estimator object.
    """

    if xml_parameters.optimization_method_type == 'GradientAscent'.lower():
        estimator = GradientAscent()
        estimator.initial_step_size = xml_parameters.initial_step_size
        estimator.max_line_search_iterations = xml_parameters.max_line_search_iterations
        estimator.line_search_shrink = xml_parameters.line_search_shrink
        estimator.line_search_expand = xml_parameters.line_search_expand

    elif xml_parameters.optimization_method_type == 'ScipyLBFGS'.lower():
        estimator = ScipyOptimize()
        estimator.max_line_search_iterations = xml_parameters.max_line_search_iterations
        estimator.memory_length = xml_parameters.memory_length
        if not model.is_frozen['template_data'] and model.use_sobolev_gradient and estimator.memory_length > 1:
            estimator.memory_length = 1
            msg = 'Impossible to use a Sobolev gradient for the template data with the ScipyLBFGS estimator memory ' \
                  'length being larger than 1. Overriding the "memory_length" option, now set to "1".'
            warnings.warn(msg)

    elif xml_parameters.optimization_method_type == 'McmcSaem'.lower():
        sampler = SrwMhwgSampler()

        momenta_proposal_distribution = MultiScalarNormalDistribution()
        # initial_control_points = model.get_control_points()
        # momenta_proposal_distribution.set_mean(np.zeros(initial_control_points.size,))
        momenta_proposal_distribution.set_variance_sqrt(xml_parameters.momenta_proposal_std)
        sampler.individual_proposal_distributions['momenta'] = momenta_proposal_distribution

        estimator = McmcSaem()
        estimator.sampler = sampler

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
    estimator.convergence_tolerance = xml_parameters.convergence_tolerance

    estimator.print_every_n_iters = xml_parameters.print_every_n_iters
    estimator.save_every_n_iters = xml_parameters.save_every_n_iters

    estimator.dataset = dataset
    estimator.statistical_model = model

    # Initial random effects realizations ------------------------------------------------------------------------------
    estimator.individual_RER['sources'] = sources
    estimator.individual_RER['onset_age'] = onset_ages
    estimator.individual_RER['log_acceleration'] = log_accelerations

    """
    Launch.
    """

    if not os.path.exists(Settings().output_dir): os.makedirs(Settings().output_dir)

    model.name = 'LongitudinalAtlas'
    print('')
    print('[ estimator.update() method ]')
    print('')

    start_time = time.time()
    estimator.update()
    end_time = time.time()
    print('>> Estimation took: ' + str(time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))))
