#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import argparse
import logging

from api.deformetrica import Deformetrica
from core import default
import support.kernels as kernel_factory
from in_out.xml_parameters import XmlParameters
from launch.compute_parallel_transport import compute_parallel_transport
from launch.estimate_bayesian_atlas import estimate_bayesian_atlas
from launch.estimate_rigid_atlas import estimate_rigid_atlas
from launch.estimate_geodesic_regression import estimate_geodesic_regression
from launch.estimate_longitudinal_atlas import estimate_longitudinal_atlas
from launch.estimate_longitudinal_metric_model import estimate_longitudinal_metric_model
from launch.estimate_longitudinal_metric_registration import estimate_longitudinal_metric_registration
from launch.estimate_longitudinal_registration import estimate_longitudinal_registration
from launch.compute_shooting import run_shooting
from in_out.dataset_functions import create_dataset


logger = logging.getLogger(__name__)


def info():
    version = open(os.path.dirname(os.path.realpath(__file__)) + '/../VERSION').read()
    return """
    ##############################
    ##### Deformetrica {version} #####
    ##############################
    """.format(version=version)


def main():
    logger_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # parse arguments
    parser = argparse.ArgumentParser(description='Deformetrica')
    parser.add_argument('model', type=str, help='model xml file')
    parser.add_argument('dataset', type=str, help='data-set xml file')
    parser.add_argument('optimization', type=str, help='optimization parameters xml file')

    # optional arguments
    parser.add_argument('-o', '--output', type=str, help='output folder')
    # logging levels: https://docs.python.org/2/library/logging.html#logging-levels
    parser.add_argument('--verbosity', '-v',
                        type=str,
                        default='WARNING',
                        choices=['NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='set output verbosity')

    args = parser.parse_args()

    # set logging level
    try:
        log_level = logging.getLevelName(args.verbosity)
        logging.basicConfig(level=log_level, format=logger_format)
    except ValueError:
        logger.warning('Logging level was not recognized. Using INFO.')
        log_level = logging.INFO

    logger.debug('Using verbosity level: ' + args.verbosity)
    logging.basicConfig(level=log_level, format=logger_format)

    # Basic info printing
    logger.info(info())

    """
    Read xml files, set general settings, and call the adapted function.
    """
    try:
        if args.output is None:
            output_dir = default.output_dir
            logger.info('No output directory defined, using default: ' + output_dir)
            os.makedirs(output_dir)
        else:
            logger.info('Setting output directory to: ' + args.output)
            output_dir = args.output
    except FileExistsError:
        pass

    deformetrica = Deformetrica(output_dir=output_dir)

    file_handler = logging.FileHandler(os.path.join(output_dir, 'log.txt'), mode='w')
    logger.addHandler(file_handler)

    logger.info('[ read_all_xmls function ]')
    xml_parameters = XmlParameters()
    xml_parameters.read_all_xmls(args.model, args.dataset, args.optimization, output_dir)

    dataset = create_dataset(
        xml_parameters.dataset_filenames,
        xml_parameters.visit_ages,
        xml_parameters.subject_ids,
        xml_parameters.template_specifications,
        dimension=xml_parameters.dimension,
        tensor_scalar_type=xml_parameters.tensor_scalar_type)

    if xml_parameters.model_type == 'DeterministicAtlas'.lower() or xml_parameters.model_type == 'Registration'.lower():
        deformetrica.estimate_deterministic_atlas(xml_parameters.template_specifications, dataset,
                                                  estimator=__get_estimator_class(xml_parameters),
                                                  estimator_options=__get_estimator_options(xml_parameters),
                                                  deformation_kernel=kernel_factory.factory(
                                                      xml_parameters.deformation_kernel_type,
                                                      kernel_width=xml_parameters.deformation_kernel_width),
                                                  **__get_run_options(xml_parameters))

    elif xml_parameters.model_type == 'BayesianAtlas'.lower():
        deformetrica.estimate_bayesian_atlas(xml_parameters.template_specifications, dataset,
                                             estimator=__get_estimator_class(xml_parameters),
                                                  estimator_options=__get_estimator_options(xml_parameters),
                                                  deformation_kernel=kernel_factory.factory(
                                                      xml_parameters.deformation_kernel_type,
                                                      kernel_width=xml_parameters.deformation_kernel_width),
                                                  **__get_run_options(xml_parameters))

    elif xml_parameters.model_type == 'RigidAtlas'.lower():
        estimate_rigid_atlas(xml_parameters)

    elif xml_parameters.model_type == 'Regression'.lower():
        estimate_geodesic_regression(xml_parameters)

    elif xml_parameters.model_type == 'LongitudinalAtlas'.lower():
        estimate_longitudinal_atlas(xml_parameters)

    elif xml_parameters.model_type == 'LongitudinalRegistration'.lower():
        estimate_longitudinal_registration(xml_parameters)

    elif xml_parameters.model_type == 'Shooting'.lower():
        run_shooting(xml_parameters)

    elif xml_parameters.model_type == 'ParallelTransport'.lower():
        compute_parallel_transport(xml_parameters)

    elif xml_parameters.model_type == 'LongitudinalMetricLearning'.lower():
        estimate_longitudinal_metric_model(xml_parameters)

    elif xml_parameters.model_type == 'LongitudinalMetricRegistration'.lower():
        estimate_longitudinal_metric_registration(xml_parameters)

    else:
        raise RuntimeError('Unrecognized model-type: "' + xml_parameters.model_type
                           + '". Check the corresponding field in the model.xml input file.')


def __get_estimator_class(xml_parameters):
    from core.estimators.gradient_ascent import GradientAscent
    from core.estimators.scipy_optimize import ScipyOptimize

    estimator = GradientAscent
    if xml_parameters.optimization_method_type.lower() == 'GradientAscent'.lower():
        estimator = GradientAscent
    elif xml_parameters.optimization_method_type.lower() == 'ScipyLBFGS'.lower():
        estimator = ScipyOptimize

    return estimator


def __get_estimator_options(xml_parameters):
    options = {}

    if xml_parameters.optimization_method_type.lower() == 'GradientAscent'.lower():
        options['initial_step_size'] = xml_parameters.initial_step_size
        options['scale_initial_step_size'] = xml_parameters.scale_initial_step_size
        options['line_search_shrink'] = xml_parameters.line_search_shrink
        options['line_search_expand'] = xml_parameters.line_search_expand

    elif xml_parameters.optimization_method_type.lower() == 'ScipyLBFGS'.lower():
        options['memory_length'] = xml_parameters.memory_length
        options['freeze_template'] = xml_parameters.freeze_template
        options['use_sobolev_gradient'] = xml_parameters.use_sobolev_gradient
        if not xml_parameters.freeze_template and xml_parameters.use_sobolev_gradient and xml_parameters.memory_length > 1:
            print('>> Using a Sobolev gradient for the template data with the ScipyLBFGS estimator memory length '
                  'being larger than 1. Beware: that can be tricky.')

    # common options
    options['max_iterations'] = xml_parameters.max_iterations
    options['max_line_search_iterations'] = xml_parameters.max_line_search_iterations
    options['convergence_tolerance'] = xml_parameters.convergence_tolerance
    options['print_every_n_iters'] = xml_parameters.print_every_n_iters
    options['save_every_n_iters'] = xml_parameters.save_every_n_iters
    options['optimized_log_likelihood'] = xml_parameters.optimized_log_likelihood

    logger.debug(options)

    return options


def __get_run_options(xml_parameters):
    options = {'number_of_time_points': xml_parameters.number_of_time_points,
               'use_rk2_for_shoot': xml_parameters.use_rk2_for_shoot,
               'use_rk2_for_flow': xml_parameters.use_rk2_for_flow,
               'freeze_template': xml_parameters.freeze_template,
               'freeze_control_points': xml_parameters.freeze_control_points,
               'use_sobolev_gradient': xml_parameters.use_sobolev_gradient,
               'smoothing_kernel_width': xml_parameters.deformation_kernel_width * xml_parameters.sobolev_kernel_width_ratio,
               'initial_control_points': xml_parameters.initial_control_points,
               'initial_cp_spacing': xml_parameters.initial_cp_spacing,
               'initial_momenta': xml_parameters.initial_momenta,
               # 'ignore_noise_variance': xml_parameters.ignore_noise_variance,
               'dense_mode': xml_parameters.dense_mode,
               'number_of_threads': xml_parameters.number_of_threads}

    logger.debug(options)

    return options


if __name__ == "__main__":
    # execute only if run as a script
    main()
