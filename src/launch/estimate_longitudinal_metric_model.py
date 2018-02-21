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

from pydeformetrica.src.support.utilities.general_settings import Settings
from pydeformetrica.src.core.model_tools.manifolds.generic_geodesic import GenericGeodesic
from pydeformetrica.src.core.model_tools.manifolds.one_dimensional_exponential import OneDimensionalExponential
from pydeformetrica.src.core.model_tools.manifolds.exponential_factory import ExponentialFactory
from pydeformetrica.src.core.models.one_dimensional_metric_learning import OneDimensionalMetricLearning
from pydeformetrica.src.in_out.dataset_functions import create_scalar_dataset


import matplotlib.pyplot as plt


# Ask Igor the format of the input !


def instantiate_longitudinal_metric_model(xml_parameters, dataset):
    model = OneDimensionalMetricLearning()

    # Factory for the manifold exponential::
    exponential_factory = ExponentialFactory()
    exponential_factory.set_manifold_type("one_dimensional")


    # Reference time
    model.set_reference_time(xml_parameters.t0)

    # Initial velocity
    model.set_v0(xml_parameters.v0)

    # Initial position
    model.set_p0(xml_parameters.p0)

    #Initial guess for the metric parameters
    model.number_of_interpolation_points = 20
    model.set_metric_parameters(np.ones(model.number_of_interpolation_points,)/model.number_of_interpolation_points)

    # Parameters of the manifold:
    manifold_parameters = {}
    manifold_parameters['number_of_interpolation_points'] = model.number_of_interpolation_points
    manifold_parameters['width'] = 1.5/model.number_of_interpolation_points
    manifold_parameters['interpolation_points_torch'] = Variable(torch.from_numpy(np.linspace(-0.2, 1., model.number_of_interpolation_points))
                                                                 .type(Settings().tensor_scalar_type),
                                                                 requires_grad=False)
    manifold_parameters['interpolation_values_torch'] = Variable(torch.from_numpy(model.get_metric_parameters())
                                                                 .type(Settings().tensor_scalar_type))
    exponential_factory.set_parameters(manifold_parameters)

    model.geodesic = GenericGeodesic(exponential_factory)
    model.geodesic.set_concentration_of_time_points(xml_parameters.concentration_of_time_points)

    #Time shift variance
    model.set_onset_age_variance(xml_parameters.initial_time_shift_variance)

    #Log acceleration variance
    model.set_log_acceleration_variance(xml_parameters.initial_log_acceleration_variance)

    number_of_subjects = dataset.number_of_subjects
    total_number_of_observations = dataset.total_number_of_observations

    # Initializations of the individual random effects

    # onset_ages: we initialize them to tau_i = t_baseline_i + 2
    onset_ages = np.zeros((number_of_subjects,))
    for i in range(number_of_subjects):
        onset_ages[i] = dataset.times[i][0].data.numpy()[0] + 2.
    # onset_ages += model.get_reference_time()
    model.set_reference_time(np.mean(onset_ages))

    log_accelerations = np.zeros((number_of_subjects))

    individual_RER = {}
    individual_RER['onset_age'] = onset_ages
    individual_RER['log_acceleration'] = log_accelerations

    model.update()
    initial_noise_variance = model.get_noise_variance()

    if initial_noise_variance is None:

        v0, p0, metric_parameters = model._fixed_effects_to_torch_tensors(False)
        p0.requires_grad = True
        onset_ages, log_accelerations = model._individual_RER_to_torch_tensors(individual_RER, False)

        residuals = model._compute_residuals(dataset, v0, p0, log_accelerations, onset_ages, metric_parameters)

        total_residual = 0.
        for i in range(len(residuals)):
            for j in range(len(residuals[i])):
                total_residual += residuals[i][j].data.numpy()[0]

        dof = total_number_of_observations
        nv = 0.0001 * total_residual / dof

        model.priors['noise_variance'].degrees_of_freedom.append(dof)
        model.priors['noise_variance'].scale_scalars.append(nv)
        model.set_noise_variance(nv)
        print("A first residual evaluation yields a noise variance of ", nv, "used for the prior")


    return model, individual_RER


def estimate_longitudinal_metric_model(xml_parameters):
    print('')
    print('[ estimate_longitudinal_metric_model function ]')
    print('')

    dataset = create_scalar_dataset(xml_parameters)

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

    # elif xml_parameters.optimization_method_type == 'McmcSaem'.lower():
    #     sampler = SrwMhwgSampler()
    #
    #     momenta_proposal_distribution = MultiScalarNormalDistribution()
    #     # initial_control_points = model.get_control_points()
    #     # momenta_proposal_distribution.set_mean(np.zeros(initial_control_points.size,))
    #     momenta_proposal_distribution.set_variance_sqrt(xml_parameters.momenta_proposal_std)
    #     sampler.individual_proposal_distributions['momenta'] = momenta_proposal_distribution
    #
    #     estimator = McmcSaem()
    #     estimator.sampler = sampler

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

    model.name = 'LongitudinalAtlas'
    print('')
    print('[ update method of the ' + estimator.name + ' optimizer ]')

    start_time = time.time()
    estimator.update()
    estimator.write()
    end_time = time.time()
    print('>> Estimation took: ' + str(time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))))




# def plot_points(l, times=None):
#     l_numpy = [elt.data.numpy() for elt in l]
#     if times is not None:
#         t = times
#     else:
#         t = np.linspace(0., 1., len(l_numpy))
#     plt.plot(t, l_numpy)
#
# # We would like to try the geodesic object on one_dim_manifold !
# exponential_factory = ExponentialFactory()
# exponential_factory.set_manifold_type("one_dimensional")
#
# manifold_parameters = {}
# manifold_parameters['number_of_interpolation_points'] = 20
# manifold_parameters['width'] = 0.1/20
# manifold_parameters['interpolation_points_torch'] = Variable(torch.from_numpy(np.linspace(0, 1, 20))
#                                                              .type(Settings().tensor_scalar_type))
# manifold_parameters['interpolation_values_torch'] = Variable(torch.from_numpy(np.random.binomial(2, 0.5, 20))
#                                                              .type(Settings().tensor_scalar_type))
# exponential_factory.set_parameters(manifold_parameters)
#
# generic_geodesic = GenericGeodesic(exponential_factory)
#
# generic_geodesic.set_t0(0.)
# generic_geodesic.set_tmin(-1.)
# generic_geodesic.set_tmax(1.)
#
# generic_geodesic.set_concentration_of_time_points(50)
#
#
# q0 = 0.5
# v0 = 0.5
# p0 = v0
#
# q = Variable(torch.Tensor([q0]), requires_grad=True).type(torch.DoubleTensor)
# p = Variable(torch.Tensor([p0]), requires_grad=False).type(torch.DoubleTensor)
#
# generic_geodesic.set_position_t0(q)
# generic_geodesic.set_momenta_t0(p)
#
#
# for i in range(10):
#     interp_values = Variable(torch.from_numpy(np.random.binomial(2, 0.5, 20))
#                                                              .type(Settings().tensor_scalar_type))
#     generic_geodesic.set_parameters(interp_values)
#     generic_geodesic.update()
#
#     traj = generic_geodesic._get_geodesic_trajectory()
#
#
#     plot_points(traj)
# plt.show()
#
#
