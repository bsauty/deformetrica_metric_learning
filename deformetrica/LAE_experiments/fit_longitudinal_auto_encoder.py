#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import argparse
import logging
import torch
import sys

from copy import deepcopy

from deformetrica.core.estimator_tools.samplers.srw_mhwg_sampler import SrwMhwgSampler
from deformetrica.core.estimators.gradient_ascent import GradientAscent
from deformetrica.core.estimators.mcmc_saem import McmcSaem
# Estimators
from deformetrica.core.estimators.scipy_optimize import ScipyOptimize
from deformetrica.core.model_tools.manifolds.exponential_factory import ExponentialFactory
from deformetrica.core.model_tools.manifolds.generic_spatiotemporal_reference_frame import GenericSpatiotemporalReferenceFrame
from deformetrica.core.models.longitudinal_metric_learning import LongitudinalMetricLearning
from deformetrica.core.models.model_functions import create_regular_grid_of_points
from deformetrica.in_out.array_readers_and_writers import *
from deformetrica.in_out.dataset_functions import create_image_dataset_from_torch
from deformetrica.support.probability_distributions.multi_scalar_normal_distribution import MultiScalarNormalDistribution
from deformetrica.support.utilities.general_settings import Settings
from deformetrica.core.models import LongitudinalAutoEncoder

sys.path.append('/home/benoit.sautydechalon/deformetrica')
import deformetrica as dfca
from deformetrica.support.utilities.general_settings import Settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def initialize_spatiotemporal_reference_frame(model, xml_parameters, logger, observation_type='image'):
    """
    Initialize everything which is relative to the geodesic its parameters.
    """
    assert xml_parameters.dimension is not None, "Provide a dimension for the longitudinal metric learning atlas."

    exponential_factory = ExponentialFactory()
    exponential_factory.set_manifold_type('euclidian')
    logger.info('Initialized the Euclidian metric for latent space')

    model.spatiotemporal_reference_frame = GenericSpatiotemporalReferenceFrame(exponential_factory)
    model.spatiotemporal_reference_frame.set_concentration_of_time_points(xml_parameters.concentration_of_time_points)
    model.spatiotemporal_reference_frame.set_number_of_time_points(xml_parameters.number_of_time_points)

    model.no_parallel_transport = False
    model.spatiotemporal_reference_frame.no_parallel_transport = False
    model.number_of_sources = xml_parameters.number_of_sources

def instantiate_longitudinal_auto_encoder_model(logger, dataset=None, xml_parameters=None, number_of_subjects=None, observation_type='image'):

    model = LongitudinalAutoEncoder()
    model.observation_type = 'image'

    if dataset is not None:
        template = dataset.deformable_objects[0][0] # because we only care about its 'metadata'
        model.template = deepcopy(template)

    # Initialize the fixed effects, either from files or to arbitrary values
    if xml_parameters is not None:
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
        model.is_frozen['onset_age_variance'] = xml_parameters.freeze_onset_age_variance
        # Log acceleration variance
        model.set_log_acceleration_variance(xml_parameters.initial_acceleration_variance)
        model.is_frozen["log_acceleration_variance"] = xml_parameters.freeze_acceleration_variance
        # Noise variance
        model.set_noise_variance(xml_parameters.initial_noise_variance)
        # Modulation matrix
        modulation_matrix = read_2D_array(xml_parameters.initial_modulation_matrix)
        if len(modulation_matrix.shape) == 1:
            modulation_matrix = modulation_matrix.reshape(Settings().dimension, 1)
        logger.info(f">> Reading {str(modulation_matrix.shape[1]) }-source initial modulation matrix from file: {xml_parameters.initial_modulation_matrix}")
        assert xml_parameters.number_of_sources == modulation_matrix.shape[1], "Please set correctly the number of sources"
        model.set_modulation_matrix(modulation_matrix)
        model.number_of_sources = modulation_matrix.shape[1]

    else:
        model.set_reference_time(70)
        model.set_v0(np.ones(Settings().dimension))
        model.set_p0(np.zeros(Settings().dimension))
        model.set_onset_age_variance(15)
        model.set_log_acceleration_variance(0.1)
        model.number_of_sources = xml_parameters.number_of_sources
        modulation_matrix = np.zeros((Settings().dimension, model.number_of_sources))
        model.set_modulation_matrix(modulation_matrix)
        model.initialize_modulation_matrix_variables()

    # Initializations of the individual random effects
    assert not (dataset is None and number_of_subjects is None), "Provide at least one info"

    if dataset is not None:
        number_of_subjects = dataset.number_of_subjects

    # Initialization the individual parameters
    if xml_parameters is not None:
        logger.info(f"Setting initial onset ages from {xml_parameters.initial_onset_ages} file")
        onset_ages = read_2D_array(xml_parameters.initial_onset_ages).reshape((len(dataset.times),))
        logger.info(f"Setting initial log accelerations from { xml_parameters.initial_accelerations} file")
        log_accelerations = read_2D_array(xml_parameters.initial_accelerations).reshape((len(dataset.times),))

    else:
        logger.info("Initializing all the onset_ages to the reference time.")
        onset_ages = np.zeros((number_of_subjects,))
        onset_ages += model.get_reference_time()
        logger.info("Initializing all log-accelerations to zero.")
        log_accelerations = np.zeros((number_of_subjects,))

    individual_RER = {}
    individual_RER['onset_age'] = onset_ages
    individual_RER['log_acceleration'] = log_accelerations

    # Initialization of the spatiotemporal reference frame.
    initialize_spatiotemporal_reference_frame(model, xml_parameters, dataset, logger, observation_type=observation_type)


    # Sources initialization
    if xml_parameters.initial_sources is not None:
        logger.info(f"Setting initial sources from {xml_parameters.initial_sources} file")
        individual_RER['sources'] = read_2D_array(xml_parameters.initial_sources).reshape(len(dataset.times), model.number_of_sources)

    elif model.number_of_sources > 0:
        # Actually here we initialize the sources to almost zero values to avoid renormalization issues (div 0)
        logger.info("Initializing all sources to zero")
        individual_RER['sources'] = np.random.normal(0,0.1,(number_of_subjects, model.number_of_sources))
    model.initialize_source_variables()

    if dataset is not None:
        total_number_of_observations = dataset.total_number_of_observations
        model.number_of_subjects = dataset.number_of_subjects

        if model.get_noise_variance() is None:

            v0, p0, metric_parameters, modulation_matrix = model._fixed_effects_to_torch_tensors(False)
            onset_ages, log_accelerations, sources = model._individual_RER_to_torch_tensors(individual_RER, False)

            residuals = model._compute_residuals(dataset, v0, p0, metric_parameters, modulation_matrix,
                                            log_accelerations, onset_ages, sources)

            total_residual = 0.
            for i in range(len(residuals)):
                total_residual += torch.sum(residuals[i]).cpu().data.numpy()

            dof = total_number_of_observations
            nv = total_residual / dof
            model.set_noise_variance(nv)
            logger.info(f">> Initial noise variance set to {nv} based on the initial mean residual value.")

        if not model.is_frozen['noise_variance']:
            dof = total_number_of_observations
            model.priors['noise_variance'].degrees_of_freedom.append(dof)

    else:
        if model.get_noise_variance() is None:
            raise RuntimeError("I can't initialize the initial noise variance: no dataset and no initialization given.")

    model.is_frozen['noise_variance'] = xml_parameters.freeze_noise_variance

    model.update()

    return model, individual_RER


def estimate_longitudinal_auto_encoder_model(data, logger):
    logger.info('')
    logger.info('[ estimate_longitudinal_metric_model function ]')

    dataset = create_image_dataset_from_torch(data)
    model, individual_RER = instantiate_longitudinal_auto_encoder_model(logger, dataset)

    sampler = SrwMhwgSampler()
    estimator = McmcSaem(model, dataset, 'McmcSaem', individual_RER, max_iterations=xml_parameters.max_iterations,
             print_every_n_iters=1, save_every_n_iters=10)
    estimator.sampler = sampler

    # Onset age proposal distribution.
    onset_age_proposal_distribution = MultiScalarNormalDistribution()
    onset_age_proposal_distribution.set_variance_sqrt(xml_parameters.onset_age_proposal_std)
    sampler.individual_proposal_distributions['onset_age'] = onset_age_proposal_distribution

    # Log-acceleration proposal distribution.
    log_acceleration_proposal_distribution = MultiScalarNormalDistribution()
    log_acceleration_proposal_distribution.set_variance_sqrt(xml_parameters.acceleration_proposal_std)
    sampler.individual_proposal_distributions['log_acceleration'] = log_acceleration_proposal_distribution

    # Sources proposal distribution

    if model.number_of_sources > 0:
        sources_proposal_distribution = MultiScalarNormalDistribution()
        # Here we impose the sources variance to be 1
        sources_proposal_distribution.set_variance_sqrt(1)
        #sources_proposal_distribution.set_variance_sqrt(xml_parameters.sources_proposal_std)
        sampler.individual_proposal_distributions['sources'] = sources_proposal_distribution

    estimator.sample_every_n_mcmc_iters = xml_parameters.sample_every_n_mcmc_iters
    estimator._initialize_acceptance_rate_information()

    # Gradient-based estimator.
    # TODO : update the LAE

    estimator.convergence_tolerance = xml_parameters.convergence_tolerance

    estimator.print_every_n_iters = xml_parameters.print_every_n_iters
    estimator.save_every_n_iters = xml_parameters.save_every_n_iters

    estimator.dataset = dataset
    estimator.statistical_model = model

    # Initial random effects realizations
    estimator.individual_RER = individual_RER

    if not os.path.exists(Settings().output_dir): os.makedirs(Settings().output_dir)

    model.name = 'LongitudinalMetricModel'
    logger.info('')
    logger.info(f"[ update method of the {estimator.name}  optimizer ]")

    start_time = time.time()
    estimator.update()
    estimator.write()
    end_time = time.time()
    logger.info(f">> Estimation took: {end_time-start_time}")


data = torch.load('mini_dataset')
Settings().dimension = 10
Settings().number_of_sources = 4
estimate_longitudinal_auto_encoder_model(data, logger)
print('ok')