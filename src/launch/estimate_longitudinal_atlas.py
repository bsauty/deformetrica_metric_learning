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
from src.in_out.array_readers_and_writers import *


def instantiate_longitudinal_atlas_model(xml_parameters, dataset=None, ignore_noise_variance=False):
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
        momenta = read_3D_array(xml_parameters.initial_momenta)
        print('>> Reading ' + str(len(control_points)) + ' initial momenta from file: '
              + xml_parameters.initial_momenta)
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
    number_of_subjects = len(xml_parameters.dataset_filenames)
    total_number_of_observations = sum([len(elt) for elt in xml_parameters.dataset_filenames])

    # Onset ages.
    if xml_parameters.initial_onset_ages is not None:
        onset_ages = read_2D_array(xml_parameters.initial_onset_ages)
        print('>> Reading initial onset ages from file: ' + xml_parameters.initial_onset_ages)
    else:
        onset_ages = np.zeros((number_of_subjects,)) + model.get_reference_time()
        print('>> Initializing all onset ages to the initial reference time: %.2f' % model.get_reference_time())

    # Log-accelerations.
    if xml_parameters.initial_log_accelerations is not None:
        log_accelerations = read_2D_array(xml_parameters.initial_log_accelerations)
        print('>> Reading initial log-accelerations from file: ' + xml_parameters.initial_log_accelerations)
    else:
        log_accelerations = np.zeros((number_of_subjects,))
        print('>> Initializing all log-accelerations to zero.')

    # Sources.
    if xml_parameters.initial_sources is not None:
        sources = read_2D_array(xml_parameters.initial_sources).reshape((-1, model.number_of_sources))
        print('>> Reading initial sources from file: ' + xml_parameters.initial_sources)
    else:
        sources = np.zeros((number_of_subjects, model.number_of_sources))
        print('>> Initializing all sources to zero')

    # Final gathering.
    individual_RER = {}
    individual_RER['sources'] = sources
    individual_RER['onset_age'] = onset_ages
    individual_RER['log_acceleration'] = log_accelerations

    # Special case of the noise variance -------------------------------------------------------------------------------
    model.is_frozen['noise_variance'] = xml_parameters.freeze_noise_variance
    initial_noise_variance = model.get_noise_variance()

    # Compute residuals if needed.
    if not ignore_noise_variance:

        # Compute initial residuals if needed.
        if np.min(initial_noise_variance) < 0:

            template_data, control_points, momenta, modulation_matrix = model._fixed_effects_to_torch_tensors(False)
            sources, onset_ages, log_accelerations = model._individual_RER_to_torch_tensors(individual_RER, False)
            residuals = model._compute_residuals(dataset, template_data, control_points, momenta, modulation_matrix,
                                                 sources, onset_ages, log_accelerations)

            residuals_per_object = np.zeros((model.number_of_objects,))
            for i in range(len(residuals)):
                for j in range(len(residuals[i])):
                    residuals_per_object += residuals[i][j].data.numpy()

            # Initialize noise variance fixed effect, and the noise variance prior if needed.
            for k, obj in enumerate(xml_parameters.template_specifications.values()):
                dof = total_number_of_observations * obj['noise_variance_prior_normalized_dof'] * \
                      model.objects_noise_dimension[k]
                nv = 0.01 * residuals_per_object[k] / dof

                if initial_noise_variance[k] < 0:
                    print('>> Initial noise variance set to %.2f based on the initial mean residual value.' % nv)
                    model.objects_noise_variance[k] = nv

        # Initialize the dof if needed.
        if not model.is_frozen['noise_variance']:
            for k, obj in enumerate(xml_parameters.template_specifications.values()):
                dof = total_number_of_observations * obj['noise_variance_prior_normalized_dof'] * \
                      model.objects_noise_dimension[k]
                model.priors['noise_variance'].degrees_of_freedom.append(dof)

    # Final initialization steps by the model object itself ------------------------------------------------------------
    model.update()

    return model, individual_RER


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

    model, individual_RER = instantiate_longitudinal_atlas_model(xml_parameters, dataset)

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
            print('>> Using a Sobolev gradient for the template data with the ScipyLBFGS estimator memory length '
                  'being larger than 1. Beware: that can be tricky.')
            # estimator.memory_length = 1
            # msg = 'Impossible to use a Sobolev gradient for the template data with the ScipyLBFGS estimator memory ' \
            #       'length being larger than 1. Overriding the "memory_length" option, now set to "1".'
            # warnings.warn(msg)

    elif xml_parameters.optimization_method_type == 'McmcSaem'.lower():
        sampler = SrwMhwgSampler()

        # Onset age proposal distribution.
        onset_age_proposal_distribution = MultiScalarNormalDistribution()
        onset_age_proposal_distribution.set_variance_sqrt(xml_parameters.onset_age_proposal_std)
        sampler.individual_proposal_distributions['onset_age'] = onset_age_proposal_distribution

        # Log-acceleration proposal distribution.
        log_acceleration_proposal_distribution = MultiScalarNormalDistribution()
        log_acceleration_proposal_distribution.set_variance_sqrt(xml_parameters.log_acceleration_proposal_std)
        sampler.individual_proposal_distributions['log_acceleration'] = log_acceleration_proposal_distribution

        # Sources proposal distribution.
        sources_proposal_distribution = MultiScalarNormalDistribution()
        sources_proposal_distribution.set_variance_sqrt(xml_parameters.sources_proposal_std)
        sampler.individual_proposal_distributions['sources'] = sources_proposal_distribution

        estimator = McmcSaem()
        estimator.sampler = sampler
        estimator.maximize_every_n_iters = xml_parameters.maximize_every_n_iters
        estimator.print_every_n_iters = xml_parameters.print_every_n_iters

    else:
        estimator = GradientAscent()
        estimator.initial_step_size = xml_parameters.initial_step_size
        estimator.max_line_search_iterations = xml_parameters.max_line_search_iterations
        estimator.line_search_shrink = xml_parameters.line_search_shrink
        estimator.line_search_expand = xml_parameters.line_search_expand

        msg = 'Unknown optimization-method-type: \"' + xml_parameters.optimization_method_type \
              + '\". Defaulting to GradientAscent.'
        warnings.warn(msg)

    estimator.optimized_log_likelihood = xml_parameters.optimized_log_likelihood

    estimator.max_iterations = xml_parameters.max_iterations
    estimator.convergence_tolerance = xml_parameters.convergence_tolerance

    estimator.print_every_n_iters = xml_parameters.print_every_n_iters
    estimator.save_every_n_iters = xml_parameters.save_every_n_iters

    estimator.dataset = dataset
    estimator.statistical_model = model

    # Initial random effects realizations
    estimator.individual_RER = individual_RER

    """
    Launch.
    """

    if not os.path.exists(Settings().output_dir): os.makedirs(Settings().output_dir)

    model.name = 'LongitudinalAtlas'
    print('')
    print('[ update method of the ' + estimator.name + ' optimizer ]')
    print('')

    start_time = time.time()
    estimator.update()
    estimator.write()
    end_time = time.time()
    print('>> Estimation took: ' + str(time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))))

    return model
