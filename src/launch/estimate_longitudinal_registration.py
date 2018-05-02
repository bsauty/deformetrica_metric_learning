import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

import warnings
import time
import shutil
from multiprocessing import Pool

from pydeformetrica.src.launch.estimate_longitudinal_atlas import instantiate_longitudinal_atlas_model
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


def estimate_longitudinal_registration_for_subject(args, overwrite=True):
    i, general_settings, xml_parameters, registration_output_path, \
    full_subject_ids, full_dataset_filenames, full_visit_ages = args

    Settings().initialize(general_settings)

    """
    Create the dataset object.
    """

    xml_parameters.dataset_filenames = [full_dataset_filenames[i]]
    xml_parameters.visit_ages = [full_visit_ages[i]]
    xml_parameters.subject_ids = [full_subject_ids[i]]

    dataset = create_dataset([full_dataset_filenames[i]], [full_visit_ages[i]], [full_subject_ids[i]],
                             xml_parameters.template_specifications)

    """
    Create a dedicated output folder for the current subject, adapt the global settings.
    """

    subject_registration_output_path = os.path.join(
        registration_output_path, 'LongitudinalRegistration__subject_' + full_subject_ids[i])

    if not overwrite and os.path.isdir(subject_registration_output_path):
        return None

    print('')
    print('[ longitudinal registration of subject ' + full_subject_ids[i] + ' ]')
    print('')

    if os.path.isdir(subject_registration_output_path):
        shutil.rmtree(subject_registration_output_path)
        os.mkdir(subject_registration_output_path)

    Settings().output_dir = subject_registration_output_path
    Settings().state_file = os.path.join(subject_registration_output_path, 'pydef_state.p')

    """
    Create the model object.
    """

    model, individual_RER = instantiate_longitudinal_atlas_model(xml_parameters, dataset)

    # In case of given initial random effect realizations, select only the relevant ones.
    for (xml_parameter, random_effect_name) \
            in zip([xml_parameters.initial_onset_ages,
                    xml_parameters.initial_log_accelerations,
                    xml_parameters.initial_sources],
                   ['onset_age', 'log_acceleration', 'sources']):
        if xml_parameter is not None and individual_RER[random_effect_name].shape[0] > 1:
            individual_RER[random_effect_name] = np.array([individual_RER[random_effect_name][i]])

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

    elif xml_parameters.optimization_method_type == 'ScipyPowell'.lower():
        estimator = ScipyOptimize()
        estimator.method = 'Powell'

    elif xml_parameters.optimization_method_type == 'McmcSaem'.lower():
        sampler = SrwMhwgSampler()

        momenta_proposal_distribution = MultiScalarNormalDistribution()
        # initial_control_points = model.get_control_points()
        # momenta_proposal_distribution.set_mean(np.zeros(initial_control_points.size,))
        momenta_proposal_distribution.set_variance_sqrt(xml_parameters.momenta_proposal_std)
        sampler.individual_proposal_distributions['momenta'] = momenta_proposal_distribution

        estimator = McmcSaem()
        estimator.sampler = sampler
        estimator.sample_every_n_mcmc_iters = xml_parameters.sample_every_n_mcmc_iters

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

    # Initial random effects realizations
    estimator.individual_RER = individual_RER

    """
    Launch.
    """

    if not os.path.exists(Settings().output_dir): os.makedirs(Settings().output_dir)

    model.name = 'LongitudinalRegistration'
    print('')
    print('[ update method of the ' + estimator.name + ' optimizer ]')

    start_time = time.time()
    estimator.update()
    model._write_model_parameters(estimator.individual_RER)
    end_time = time.time()
    print('')
    print('>> Estimation took: ' + str(time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))))

    return model


def estimate_longitudinal_registration(xml_parameters, overwrite=True):
    print('')
    print('[ estimate_longitudinal_registration function ]')

    """
    Prepare the loop over each subject.
    """

    registration_output_path = Settings().output_dir
    full_dataset_filenames = xml_parameters.dataset_filenames
    full_visit_ages = xml_parameters.visit_ages
    full_subject_ids = xml_parameters.subject_ids
    number_of_subjects = len(full_dataset_filenames)
    xml_parameters.save_every_n_iters = 100000  # Don't waste time saving intermediate results.

    # Global parameter.
    global_number_of_sources = read_2D_array(xml_parameters.initial_modulation_matrix).shape[1]

    """
    Launch the individual longitudinal registrations.
    """

    # Multi-threaded version.
    if False and Settings().number_of_threads > 1:  # It is already multi-threaded at the level below.
        pool = Pool(processes=Settings().number_of_threads)
        args = [(i, Settings().serialize(), xml_parameters, registration_output_path,
                 full_subject_ids, full_dataset_filenames, full_visit_ages)
                for i in range(number_of_subjects)]
        pool.map(estimate_longitudinal_registration_for_subject, args, overwrite)
        pool.close()
        pool.join()

    # Single thread version.
    else:
        for i in range(number_of_subjects):
            estimate_longitudinal_registration_for_subject((
                i, Settings().serialize(), xml_parameters, registration_output_path,
                full_subject_ids, full_dataset_filenames, full_visit_ages), overwrite)

    """
    Gather all the individual registration results.
    """

    print('')
    print('[ save the aggregated registration parameters of all subjects ]')
    print('')

    # Gather the individual random effect realizations.
    onset_ages = np.zeros((number_of_subjects,))
    log_accelerations = np.zeros((number_of_subjects,))
    sources = np.zeros((number_of_subjects, global_number_of_sources))

    for i in range(number_of_subjects):
        subject_registration_output_path = os.path.join(
            registration_output_path, 'LongitudinalRegistration__subject_' + full_subject_ids[i])

        onset_ages[i] = np.loadtxt(os.path.join(
            subject_registration_output_path, 'LongitudinalRegistration__EstimatedParameters__OnsetAges.txt'))
        log_accelerations[i] = np.loadtxt(os.path.join(
            subject_registration_output_path, 'LongitudinalRegistration__EstimatedParameters__LogAccelerations.txt'))
        sources[i] = np.loadtxt(os.path.join(
            subject_registration_output_path, 'LongitudinalRegistration__EstimatedParameters__Sources.txt'))

    individual_RER = {}
    individual_RER['sources'] = sources
    individual_RER['onset_age'] = onset_ages
    individual_RER['log_acceleration'] = log_accelerations

    # Write temporarily those files.
    temporary_output_path = os.path.join(registration_output_path, 'tmp')
    if os.path.isdir(temporary_output_path):
        shutil.rmtree(temporary_output_path)
    os.mkdir(temporary_output_path)

    path_to_onset_ages = os.path.join(temporary_output_path, 'onset_ages.txt')
    path_to_log_accelerations = os.path.join(temporary_output_path, 'log_acceleration.txt')
    path_to_sources = os.path.join(temporary_output_path, 'sources.txt')

    np.savetxt(path_to_onset_ages, onset_ages)
    np.savetxt(path_to_log_accelerations, log_accelerations)
    np.savetxt(path_to_sources, sources)

    # Construct the aggregated longitudinal atlas model, and save it.
    xml_parameters.dataset_filenames = full_dataset_filenames
    xml_parameters.visit_ages = full_visit_ages
    xml_parameters.subject_ids = full_subject_ids

    xml_parameters.initial_onset_ages = path_to_onset_ages
    xml_parameters.initial_log_accelerations = path_to_log_accelerations
    xml_parameters.initial_sources = path_to_sources

    Settings().output_dir = registration_output_path
    if not os.path.isdir(Settings().output_dir):
        os.mkdir(Settings().output_dir)

    dataset = create_dataset(xml_parameters.dataset_filenames, xml_parameters.visit_ages,
                             xml_parameters.subject_ids, xml_parameters.template_specifications)

    model, _ = instantiate_longitudinal_atlas_model(xml_parameters, dataset)
    model.name = 'LongitudinalRegistration'
    model.write(dataset, None, individual_RER)
