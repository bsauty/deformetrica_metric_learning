import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

import torch
from torch.autograd import Variable
import numpy as np
import warnings
import time

# Estimators
from sklearn import datasets, linear_model
from pydeformetrica.src.core.estimators.scipy_optimize import ScipyOptimize
from pydeformetrica.src.core.estimators.gradient_ascent import GradientAscent
from pydeformetrica.src.core.estimators.mcmc_saem import McmcSaem
from pydeformetrica.src.core.estimator_tools.samplers.srw_mhwg_sampler import SrwMhwgSampler
from pydeformetrica.src.support.utilities.general_settings import Settings
from pydeformetrica.src.core.model_tools.manifolds.generic_geodesic import GenericGeodesic
from pydeformetrica.src.core.model_tools.manifolds.exponential_factory import ExponentialFactory
from pydeformetrica.src.core.models.one_dimensional_metric_learning import OneDimensionalMetricLearning
from pydeformetrica.src.in_out.dataset_functions import read_and_create_scalar_dataset
from pydeformetrica.src.support.probability_distributions.multi_scalar_normal_distribution import MultiScalarNormalDistribution
from pydeformetrica.src.in_out.array_readers_and_writers import read_2D_array


def initialize_individual_effects(dataset):
    """
    least_square regression for each subject, so that yi = ai * t + bi
    output is the list of ais and bis
    this proceeds as if the initialization for the geodesic is a straight line
    """
    print("Performing initial least square regressions on the subjects, for initialization purposes.")

    number_of_subjects = dataset.number_of_subjects

    ais = []
    bis = []

    for i in range(number_of_subjects):

        # Special case of a single observation for the subject
        if len(dataset.times[i]) <= 1:
            ais.append(1.)
            bis.append(0.)

        least_squares = linear_model.LinearRegression()
        least_squares.fit(dataset.times[i].reshape(-1, 1), dataset.deformable_objects[i].data.numpy().reshape(-1, 1))

        ais.append(max(0.001, least_squares.coef_[0][0]))
        bis.append(least_squares.intercept_[0])

        #if the slope is negative, we change it to 0.03, arbitrarily...

    # Ideally replace this by longitudinal registrations on the initial metric ! (much more expensive though)

    return ais, bis

def _initialize_variables(dataset):
    ais, bis = initialize_individual_effects(dataset)
    reference_time = np.mean([np.mean(times_i) for times_i in dataset.times])
    average_a = np.mean(ais)
    average_b = np.mean(bis)
    alphas = []
    onset_ages = []
    for i in range(len(ais)):
        alphas.append(max(0.2, min(ais[i]/average_a, 2.5))) # Arbitrary bounds for a sane initialization
        onset_ages.append(max(reference_time - 15, min(reference_time + 15, (reference_time*average_a + average_b - bis[i])/ais[i])))
    # p0 = average_a * reference_time + average_b

    p0 = 0
    for i in range(dataset.number_of_subjects):
        p0 += np.mean(dataset.deformable_objects[i].data.numpy())
    p0 /= dataset.number_of_subjects

    return reference_time, average_a, p0, onset_ages, alphas

def initialize_geodesic(model, xml_parameters):
    """
    Initialize everything which is relative to the geodesic its parameters.
    """

    exponential_factory = ExponentialFactory()
    if xml_parameters.exponential_type is not None:
        print("Initializing exponential type to", xml_parameters.exponential_type)
        exponential_factory.set_manifold_type(xml_parameters.exponential_type)
    else:
        msg = "Defaulting exponential type to parametric"
        warnings.warn(msg)

    # Initial metric parameters
    if exponential_factory.manifold_type == 'parametric':
        if xml_parameters.metric_parameters_file is None:
            if xml_parameters.number_of_interpolation_points is None:
                raise ValueError("At least provide a number of interpolation points for the parametric geodesic,"
                                 " if no initial file is available")
            model.number_of_metric_parameters = xml_parameters.number_of_metric_parameters
            print("I am defaulting to the naive initialization for the parametric exponential.")
            model.set_metric_parameters(np.ones(model.number_of_interpolation_points,)/model.number_of_interpolation_points) # Starting from close to a constant metric.

        else:
            print("Setting the initial metric parameters from the",
                  xml_parameters.metric_parameters_file, "file")
            metric_parameters = np.loadtxt(xml_parameters.metric_parameters_file)
            model.number_of_interpolation_points = len(metric_parameters)
            model.set_metric_parameters(metric_parameters)

        # Parameters of the parametric manifold:
        manifold_parameters = {}
        width = 1. / model.number_of_interpolation_points
        print("The width for the metric interpolation is set to", width)
        manifold_parameters['number_of_interpolation_points'] = model.number_of_interpolation_points
        manifold_parameters['width'] = width
        manifold_parameters['interpolation_points_torch'] = Variable(
            torch.from_numpy(np.linspace(0. + width, 1. - width, model.number_of_interpolation_points))
                .type(Settings().tensor_scalar_type),
            requires_grad=False)
        manifold_parameters['interpolation_values_torch'] = Variable(torch.from_numpy(model.get_metric_parameters())
                                                                     .type(Settings().tensor_scalar_type))
        exponential_factory.set_parameters(manifold_parameters)

    elif exponential_factory.manifold_type == 'logistic':
        """ 
        No initial parameter to set ! Just freeze the model parameters (or even delete the key ?)
        """
        model.is_frozen['metric_parameters'] = True

    elif exponential_factory.manifold_type == 'fourier':
        if xml_parameters.metric_parameters_file is None:
            if xml_parameters.number_of_metric_coefficients is None:
                raise ValueError("At least provide a number of fourier coefficients for the Fourier geodesic,"
                                 " if no initial file is available")
            model.number_of_metric_parameters = xml_parameters.number_of_metric_parameters
            print("I am defaulting to the naive initialization for the fourier exponential.")
            raise ValueError("Define the naive initialization for the fourier exponential.")

        else:
            print("Setting the initial metric parameters from the",
                  xml_parameters.metric_parameters_file, "file")
            metric_parameters = np.loadtxt(xml_parameters.metric_parameters_file)
            model.number_of_interpolation_points = len(metric_parameters)
            model.set_metric_parameters(metric_parameters)

            # Parameters of the parametric manifold:
            manifold_parameters = {}
            manifold_parameters['fourier_coefficients_torch'] = Variable(torch.from_numpy(model.get_metric_parameters())
                                                                         .type(Settings().tensor_scalar_type))
            exponential_factory.set_parameters(manifold_parameters)

    model.geodesic = GenericGeodesic(exponential_factory)

    model.geodesic.set_concentration_of_time_points(xml_parameters.concentration_of_time_points)

def instantiate_longitudinal_metric_model(xml_parameters, dataset=None, number_of_subjects=None):
    model = OneDimensionalMetricLearning()

    if dataset is not None and xml_parameters.initialization_heuristic:
        reference_time, v0, p0, onset_ages, alphas = _initialize_variables(dataset)
        # Reference time
        model.set_reference_time(reference_time)
        # Initial velocity
        model.set_v0(v0)
        # Initial position
        model.set_p0(p0)
        # Time shift variance
        model.set_onset_age_variance(np.var(onset_ages))
        # Log acceleration variance
        model.set_log_acceleration_variance(np.var(alphas))

        # Noise variance
        if xml_parameters.initial_noise_variance is not None:
            model.set_noise_variance(xml_parameters.initial_noise_variance)

        # Initializations of the individual random effects

        individual_RER = {}
        individual_RER['onset_age'] = np.array(onset_ages)
        individual_RER['log_acceleration'] = np.log(alphas)

    else:
        # Reference time
        model.set_reference_time(xml_parameters.t0)
        # Initial velocity
        model.set_v0(xml_parameters.v0)
        # Initial position
        model.set_p0(xml_parameters.p0)
        # Time shift variance
        if xml_parameters.initial_time_shift_variance is not None:
            model.set_onset_age_variance(xml_parameters.initial_time_shift_variance)
        # Log acceleration variance
        if xml_parameters.initial_log_acceleration_variance is not None:
            model.set_log_acceleration_variance(xml_parameters.initial_log_acceleration_variance)
        # Noise variance
        if xml_parameters.initial_noise_variance is not None:
            model.set_noise_variance(xml_parameters.initial_noise_variance)
        # Initializations of the individual random effects
        assert not (dataset is None and number_of_subjects is None), "Provide at least one info"

        if dataset is not None:
            number_of_subjects = dataset.number_of_subjects

        onset_ages = np.zeros((number_of_subjects,))
        onset_ages += model.get_reference_time()

        log_accelerations = np.zeros((number_of_subjects,))

        individual_RER = {}
        individual_RER['onset_age'] = onset_ages
        individual_RER['log_acceleration'] = log_accelerations

    if xml_parameters.initial_onset_ages is not None:
        print("Setting initial onset ages from", xml_parameters.initial_onset_ages, "file")
        individual_RER['onset_age'] = read_2D_array(xml_parameters.initial_onset_ages)

    if xml_parameters.initial_log_accelerations is not None:
        print("Setting initial log accelerations from", xml_parameters.initial_log_accelerations, "file")
        individual_RER['log_acceleration'] = read_2D_array(xml_parameters.initial_log_accelerations)

    initialize_geodesic(model, xml_parameters)

    if dataset is not None:
        total_number_of_observations = dataset.total_number_of_observations

        if model.get_noise_variance() is None:

            v0, p0, metric_parameters = model._fixed_effects_to_torch_tensors(False)
            p0.requires_grad = True
            onset_ages, log_accelerations = model._individual_RER_to_torch_tensors(individual_RER, False)

            residuals = model._compute_residuals(dataset, v0, p0, log_accelerations, onset_ages, metric_parameters)

            total_residual = 0.
            for i in range(len(residuals)):
                total_residual += torch.sum(residuals[i]).data.numpy()[0]

            dof = total_number_of_observations
            nv = 0.01 * total_residual / dof
            model.set_noise_variance(nv)
            print('>> Initial noise variance set to %.2f based on the initial mean residual value.' % nv)

        if not model.is_frozen['noise_variance']:
            dof = total_number_of_observations
            model.priors['noise_variance'].degrees_of_freedom.append(dof)

    else:
        if model.get_noise_variance() is None:
            msg = "I can't initialize the initial noise variance: no dataset and no initialization given."
            warnings.warn(msg)

    model.update()

    return model, individual_RER


def estimate_longitudinal_metric_model(xml_parameters):
    print('')
    print('[ estimate_longitudinal_metric_model function ]')
    print('')

    dataset = read_and_create_scalar_dataset(xml_parameters)

    model, individual_RER = instantiate_longitudinal_metric_model(xml_parameters, dataset)

    if xml_parameters.optimization_method_type == 'GradientAscent'.lower():
        estimator = GradientAscent()
        estimator.initial_step_size = xml_parameters.initial_step_size
        estimator.max_line_search_iterations = xml_parameters.max_line_search_iterations
        estimator.line_search_shrink = xml_parameters.line_search_shrink
        estimator.line_search_expand = xml_parameters.line_search_expand
        estimator.scale_initial_step_size = xml_parameters.scale_initial_step_size

    elif xml_parameters.optimization_method_type == 'ScipyLBFGS'.lower():
        estimator = ScipyOptimize()
        estimator.max_line_search_iterations = xml_parameters.max_line_search_iterations
        estimator.memory_length = xml_parameters.memory_length
            # estimator.memory_length = 1
            # msg = 'Impossible to use a Sobolev gradient for the template data with the ScipyLBFGS estimator memory ' \
            #       'length being larger than 1. Overriding the "memory_length" option, now set to "1".'
            # warnings.warn(msg)

    elif xml_parameters.optimization_method_type == 'McmcSaem'.lower():
        sampler = SrwMhwgSampler()
        estimator = McmcSaem()
        estimator.sampler = sampler

        # Onset age proposal distribution.
        onset_age_proposal_distribution = MultiScalarNormalDistribution()
        onset_age_proposal_distribution.set_variance_sqrt(xml_parameters.onset_age_proposal_std)
        sampler.individual_proposal_distributions['onset_age'] = onset_age_proposal_distribution

        # Log-acceleration proposal distribution.
        log_acceleration_proposal_distribution = MultiScalarNormalDistribution()
        log_acceleration_proposal_distribution.set_variance_sqrt(xml_parameters.log_acceleration_proposal_std)
        sampler.individual_proposal_distributions['log_acceleration'] = log_acceleration_proposal_distribution
        estimator.maximize_every_n_iters = xml_parameters.maximize_every_n_iters


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

    model.name = 'LongitudinalMetricModel'
    print('')
    print('[ update method of the ' + estimator.name + ' optimizer ]')

    start_time = time.time()
    estimator.update()
    estimator.write()
    end_time = time.time()
    print('>> Estimation took: ' + str(time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))))
