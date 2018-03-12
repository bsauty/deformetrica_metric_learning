import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

import torch
from torch.autograd import Variable
import numpy as np
import warnings
import time

# Estimators
from pydeformetrica.src.core.estimators.scipy_optimize import ScipyOptimize
from pydeformetrica.src.core.estimators.gradient_ascent import GradientAscent
from pydeformetrica.src.core.estimators.mcmc_saem import McmcSaem
from pydeformetrica.src.core.estimator_tools.samplers.srw_mhwg_sampler import SrwMhwgSampler
from pydeformetrica.src.support.utilities.general_settings import Settings
from pydeformetrica.src.core.model_tools.manifolds.generic_spatiotemporal_reference_frame import GenericSpatiotemporalReferenceFrame
from pydeformetrica.src.core.model_tools.manifolds.exponential_factory import ExponentialFactory
from pydeformetrica.src.core.models.longitudinal_metric_learning import LongitudinalMetricLearning
from pydeformetrica.src.in_out.dataset_functions import read_and_create_scalar_dataset
from pydeformetrica.src.support.probability_distributions.multi_scalar_normal_distribution import MultiScalarNormalDistribution
from pydeformetrica.src.in_out.array_readers_and_writers import read_2D_array
from pydeformetrica.src.core.models.model_functions import create_regular_grid_of_points


def _initialize_parametric_exponential(model, xml_parameters, dataset, exponential_factory, metric_parameters):
    """
    if width is None: raise error
    else: if initialization_points : use them
        else: initialize them using the width
    if metric_parameters is None: naive initialization
    else: assert on the size.

    """
    dimension = Settings().dimension

    if xml_parameters.deformation_kernel_width is None:
        raise ValueError("Please provide a kernel width for the parametric exponenial")

    width = xml_parameters.deformation_kernel_width
    if xml_parameters.interpolation_points_file is None:
        print("I am initializing the interpolation points using the width", width)

    # Initializing the interpolation points.
    if xml_parameters.interpolation_points_file is None:
        box = np.zeros((Settings().dimension, 2))
        box[:, 1] = np.ones(Settings().dimension)
        interpolation_points_all = create_regular_grid_of_points(box, width)

        print("Suggested cp to fill the box:", len(interpolation_points_all))

        interpolation_points_filtered = []
        numpy_observations = np.concatenate([elt.data.numpy() for elt in dataset.deformable_objects])
        numpy_observations = numpy_observations.reshape(len(numpy_observations), dimension)
        for p in interpolation_points_all:
            if np.min(np.sum((p - numpy_observations) ** 2, 1)) < width**2:
                interpolation_points_filtered.append(p)

        print("Cp after filtering:", len(interpolation_points_filtered))
        interpolation_points = np.array(interpolation_points_filtered)

    else:
        print("Loading the interpolation points from file", xml_parameters.interpolation_points_file)
        interpolation_points = read_2D_array(xml_parameters.interpolation_points_file)

    model.number_of_interpolation_points = len(interpolation_points)

    # Now initializing the metric parameters
    if metric_parameters is None:
        # Naive initialization of the metric... with multivariate case.
        # It should be a (nb_points, dim*(dim - 1 ) /2)
        # We start by a (nb_points, dim, dim) list of upper triangular matrices
        dim = xml_parameters.dimension
        diagonal_indices = []
        spacing = 0
        pos_in_line = 0
        for j in range(int(dim*(dim+1)/2) - 1, -1, -1):
            if pos_in_line == spacing:
                spacing += 1
                pos_in_line = 0
                diagonal_indices.append(j)
            else:
                pos_in_line += 1

        metric_parameters = np.zeros((model.number_of_interpolation_points, int(dim * (dim + 1) / 2)))
        val = np.sqrt(1. / model.number_of_interpolation_points)
        for i in range(len(metric_parameters)):
            for k in range(len(metric_parameters[i])):
                if k in diagonal_indices:
                    metric_parameters[i, k] = val

    else:
        assert len(metric_parameters) == len(interpolation_points), "Bad input format for the metric parameters"
        assert len(metric_parameters[0]) == Settings().dimension * (Settings().dimension + 1)/2, "Bad input format for the metric parameters"


    # Parameters of the parametric manifold:
    manifold_parameters = {}
    print("The width for the metric interpolation is set to", width)
    manifold_parameters['number_of_interpolation_points'] = model.number_of_interpolation_points
    manifold_parameters['width'] = width

    manifold_parameters['interpolation_points_torch'] = Variable(torch.from_numpy(interpolation_points)
                                                                 .type(Settings().tensor_scalar_type),
                                                                 requires_grad=False)
    manifold_parameters['interpolation_values_torch'] = Variable(torch.from_numpy(metric_parameters)
                                                                 .type(Settings().tensor_scalar_type))
    exponential_factory.set_parameters(manifold_parameters)

    return metric_parameters

def initialize_spatiotemporal_reference_frame(model, xml_parameters, dataset):
    """
    Initialize everything which is relative to the geodesic its parameters.
    """
    assert xml_parameters.dimension is not None, "Provide a dimension for the longitudinal metric learning atlas."

    exponential_factory = ExponentialFactory()
    if xml_parameters.exponential_type is not None:
        print("Initializing exponential type to", xml_parameters.exponential_type)
        exponential_factory.set_manifold_type(xml_parameters.exponential_type)
    else:
        msg = "Defaulting exponential type to parametric"
        warnings.warn(msg)

    # Reading parameter file, if there is one:
    metric_parameters = None
    if xml_parameters.metric_parameters_file is not None:
        metric_parameters = np.sqrt(np.loadtxt(xml_parameters.metric_parameters_file))
        metric_parameters = np.reshape(metric_parameters, (len(metric_parameters), int(Settings().dimension * (Settings().dimension + 1)/2)))

    # Initial metric parameters
    if exponential_factory.manifold_type == 'parametric':
        metric_parameters = _initialize_parametric_exponential(model, xml_parameters, dataset, exponential_factory, metric_parameters)

    elif exponential_factory.manifold_type == 'logistic':
        """ 
        No initial parameter to set ! Just freeze the model parameters (or even delete the key ?)
        """
        model.is_frozen['metric_parameters'] = True

    # elif exponential_factory.manifold_type == 'fourier':
    #     if metric_parameters is None:
    #         if xml_parameters.number_of_metric_coefficients is None:
    #             raise ValueError("At least provide a number of fourier coefficients for the Fourier geodesic,"
    #                              " if no initial file is available")
    #         model.number_of_metric_parameters = xml_parameters.number_of_metric_parameters
    #         print("I am defaulting to the naive initialization for the fourier exponential.")
    #         raise ValueError("Define the naive initialization for the fourier exponential.")
    #
    #     else:
    #         print("Setting the initial metric parameters from the",
    #               xml_parameters.metric_parameters_file, "file")
    #         model.set_metric_parameters(metric_parameters)
    #
    #         # Parameters of the parametric manifold:
    #         manifold_parameters = {}
    #         manifold_parameters['fourier_coefficients_torch'] = Variable(torch.from_numpy(model.get_metric_parameters())
    #                                                                      .type(Settings().tensor_scalar_type))
    #         exponential_factory.set_parameters(manifold_parameters)

    model.spatiotemporal_reference_frame = GenericSpatiotemporalReferenceFrame(exponential_factory)
    model.spatiotemporal_reference_frame.set_concentration_of_time_points(xml_parameters.concentration_of_time_points)
    model.spatiotemporal_reference_frame.set_number_of_time_points(xml_parameters.number_of_time_points)
    model.parametric_metric = (xml_parameters.exponential_type in ['parametric'])
    if model.parametric_metric:
        model.is_frozen['metric_parameters'] = xml_parameters.freeze_metric_parameters
        model.set_metric_parameters(metric_parameters)

    if Settings().dimension == 1:
        print("I am setting the no_parallel_transport flag to True because the dimension is 1")
        model.no_parallel_transport = True
        model.spatiotemporal_reference_frame.no_parallel_transport = True
        model.number_of_sources = 0

    elif xml_parameters.number_of_sources == 0 or xml_parameters.number_of_sources is None:
        print("I am setting the no_parallel_transport flag to True because the number of sources is 0.")
        model.no_parallel_transport = True
        model.spatiotemporal_reference_frame.no_parallel_transport = True
        model.number_of_sources = 0

    else:
        print("I am setting the no_parallel_transport flag to False.")
        model.no_parallel_transport = False
        model.spatiotemporal_reference_frame.no_parallel_transport = False
        model.number_of_sources = xml_parameters.number_of_sources

def instantiate_longitudinal_metric_model(xml_parameters, dataset=None, number_of_subjects=None):
    model = LongitudinalMetricLearning()

    # Reference time
    model.set_reference_time(xml_parameters.t0)
    model.is_frozen['reference_time'] = xml_parameters.freeze_reference_time
    # Initial velocity
    initial_velocity_file = xml_parameters.v0
    model.set_v0(read_2D_array(initial_velocity_file))
    model.is_frozen['v0'] = xml_parameters.freeze_v0
    # Initial position
    initial_position_file = xml_parameters.p0
    model.set_p0(read_2D_array(initial_position_file))
    model.is_frozen['p0'] = xml_parameters.freeze_p0
    # Time shift variance
    model.set_onset_age_variance(xml_parameters.initial_time_shift_variance)
    model.is_frozen['freeze_time_shift_variance'] = xml_parameters.freeze_time_shift_variance
    # Log acceleration variance
    model.set_log_acceleration_variance(xml_parameters.initial_log_acceleration_variance)
    model.is_frozen["log_acceleration_variance"] = xml_parameters.freeze_log_acceleration_variance
    # Non-mandatory parameters, the model can initialize them

    # Noise variance
    if xml_parameters.initial_noise_variance is not None:
        model.set_noise_variance(xml_parameters.initial_noise_variance)

    # Initializations of the individual random effects
    assert not (dataset is None and number_of_subjects is None), "Provide at least one info"

    if dataset is not None:
        number_of_subjects = dataset.number_of_subjects

    # Initialization from files
    if xml_parameters.initial_onset_ages is not None:
        print("Setting initial onset ages from", xml_parameters.initial_onset_ages, "file")
        onset_ages = read_2D_array(xml_parameters.initial_onset_ages)

    else:
        print("Initializing all the onset_ages to the reference time.")
        onset_ages = np.zeros((number_of_subjects,))
        onset_ages += model.get_reference_time()

    if xml_parameters.initial_log_accelerations is not None:
        print("Setting initial log accelerations from", xml_parameters.initial_log_accelerations, "file")
        log_accelerations = read_2D_array(xml_parameters.initial_log_accelerations)

    else:
        print("Initializing all log-accelerations to zero.")
        log_accelerations = np.zeros((number_of_subjects,))

    individual_RER = {}
    individual_RER['onset_age'] = onset_ages
    individual_RER['log_acceleration'] = log_accelerations

    # Initialization of the spatiotemporal reference frame.
    initialize_spatiotemporal_reference_frame(model, xml_parameters, dataset)

    # Modulation matrix.
    model.is_frozen['modulation_matrix'] = xml_parameters.freeze_modulation_matrix
    if xml_parameters.initial_modulation_matrix is not None:
        modulation_matrix = read_2D_array(xml_parameters.initial_modulation_matrix)
        print('>> Reading ' + str(modulation_matrix.shape[1]) + '-source initial modulation matrix from file: '
              + xml_parameters.initial_modulation_matrix)
        model.set_modulation_matrix(modulation_matrix)
        model.number_of_sources = modulation_matrix.shape[1]
    else:
        model.number_of_sources = xml_parameters.number_of_sources
        modulation_matrix = np.zeros((Settings().dimension, model.number_of_sources))
        model.set_modulation_matrix(modulation_matrix)
    model.initialize_modulation_matrix_variables()

    # Sources initialization
    if xml_parameters.initial_sources is not None:
        print("Setting initial sources from", xml_parameters.initial_sources, "file")
        individual_RER['sources'] = read_2D_array(xml_parameters.initial_sources)
    elif model.number_of_sources > 0:
        print("Initializing all sources to zero")
        individual_RER['sources'] = np.zeros((number_of_subjects, model.number_of_sources))
    model.initialize_source_variables()

    if dataset is not None:
        total_number_of_observations = dataset.total_number_of_observations
        model.number_of_subjects = dataset.number_of_subjects

        if model.get_noise_variance() is None:

            v0, p0, metric_parameters, modulation_matrix = model._fixed_effects_to_torch_tensors(False)
            p0.requires_grad = True
            onset_ages, log_accelerations, sources = model._individual_RER_to_torch_tensors(individual_RER, False)

            residuals = model._compute_residuals(dataset, v0, p0, metric_parameters, modulation_matrix,
                                            log_accelerations, onset_ages, sources)

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
            raise RuntimeError("I can't initialize the initial noise variance: no dataset and no initialization given.")

    model.is_frozen['noise_variance'] = xml_parameters.freeze_noise_variance

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
        estimator.scale_initial_step_size = xml_parameters.scale_initial_step_size
        estimator.max_line_search_iterations = xml_parameters.max_line_search_iterations
        estimator.line_search_shrink = xml_parameters.line_search_shrink
        estimator.line_search_expand = xml_parameters.line_search_expand


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
        estimator.sample_every_n_mcmc_iters = xml_parameters.sample_every_n_mcmc_iters

        # Gradient-based estimator.
        estimator.gradient_based_estimator = GradientAscent()
        estimator.gradient_based_estimator.statistical_model = model
        estimator.gradient_based_estimator.dataset = dataset
        estimator.gradient_based_estimator.optimized_log_likelihood = 'class2'
        estimator.gradient_based_estimator.max_iterations = 3
        estimator.gradient_based_estimator.max_line_search_iterations = 10
        estimator.gradient_based_estimator.convergence_tolerance = 1e-4
        estimator.gradient_based_estimator.print_every_n_iters = 1
        estimator.gradient_based_estimator.save_every_n_iters = 100000
        estimator.gradient_based_estimator.initial_step_size = xml_parameters.initial_step_size
        estimator.gradient_based_estimator.line_search_shrink = 0.5
        estimator.gradient_based_estimator.line_search_expand = 1.2
        estimator.gradient_based_estimator.scale_initial_step_size = True

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
