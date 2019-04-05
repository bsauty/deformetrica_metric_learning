#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import argparse
import logging
import os

import api
from __init__ import __version__
from support import utilities

from core import default
from gui.gui_window import StartGui
from in_out.xml_parameters import XmlParameters, get_dataset_specifications, get_estimator_options, get_model_options

from launch.estimate_longitudinal_metric_model import estimate_longitudinal_metric_model
from launch.estimate_longitudinal_metric_registration import estimate_longitudinal_metric_registration
from launch.initialize_longitudinal_atlas import initialize_longitudinal_atlas

logger = logging.getLogger(__name__)


def main():

    # common options
    common_parser = argparse.ArgumentParser()
    common_parser.add_argument('--parameters', '-p', type=str, help='parameters xml file')
    common_parser.add_argument('--output', '-o', type=str, help='output folder')
    # logging levels: https://docs.python.org/2/library/logging.html#logging-levels
    common_parser.add_argument('--verbosity', '-v',
                               type=str,
                               default='WARNING',
                               choices=['NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                               help='set output verbosity')

    # main parser
    description = 'Statistical analysis of 2D and 3D shape data. ' + os.linesep + os.linesep + 'version ' + __version__
    parser = argparse.ArgumentParser(prog='deformetrica', description=description, formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(title='command', dest='command')
    subparsers.required = True  # make 'command' mandatory

    # estimate command
    parser_estimate = subparsers.add_parser('estimate', add_help=False, parents=[common_parser])
    parser_estimate.add_argument('model', type=str, help='model xml file')
    parser_estimate.add_argument('dataset', type=str, help='dataset xml file')

    # compute command
    parser_compute = subparsers.add_parser('compute', add_help=False, parents=[common_parser])
    parser_compute.add_argument('model', type=str, help='model xml file')

    # initialize command
    parser_initialize = subparsers.add_parser('initialize', add_help=False, parents=[common_parser])
    parser_initialize.add_argument('model', type=str, help='model xml file')
    parser_initialize.add_argument('dataset', type=str, help='dataset xml file')
    # parser_initialize.add_argument('--overwrite', type=str, help='')

    # gui command
    subparsers.add_parser('gui', add_help=False, parents=[common_parser])

    # parser.add_argument('model', type=str, help='model xml file')
    # parser.add_argument('optimization', type=str, help='optimization parameters xml file')
    # parser.add_argument('--dataset', type=str, help='data-set xml file')

    args = parser.parse_args()

    # set logging level
    try:
        logger.setLevel(args.verbosity)
    except ValueError:
        logger.warning('Logging level was not recognized. Using INFO.')
        logger.setLevel(logging.INFO)

    if args.command == 'gui':
        StartGui().start()
        return 0
    else:

        """
        Read xml files, set general settings, and call the adapted function.
        """

        # TODO. This special case will disappear when the API interface is updated for the initializing pipelines.
        if not args.command == 'initialize':

            output_dir = None
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

            deformetrica = api.Deformetrica(output_dir=output_dir, verbosity=logger.level)

        # logger.info('[ read_all_xmls function ]')
        xml_parameters = XmlParameters()
        xml_parameters.read_all_xmls(args.model,
                                     args.dataset if args.command == 'estimate' else None,
                                     args.parameters)

        if xml_parameters.model_type == 'Registration'.lower():
            assert args.command == 'estimate', \
                'The estimation of a registration model should be launched with the command: ' \
                '"deformetrica estimate" (and not "%s").' % args.command
            deformetrica.estimate_registration(
                xml_parameters.template_specifications,
                get_dataset_specifications(xml_parameters),
                estimator_options=get_estimator_options(xml_parameters),
                model_options=get_model_options(xml_parameters))

        elif xml_parameters.model_type == 'DeterministicAtlas'.lower():
            assert args.command == 'estimate', \
                'The estimation of a deterministic atlas model should be launched with the command: ' \
                '"deformetrica estimate" (and not "%s").' % args.command
            deformetrica.estimate_deterministic_atlas(
                xml_parameters.template_specifications,
                get_dataset_specifications(xml_parameters),
                estimator_options=get_estimator_options(xml_parameters),
                model_options=get_model_options(xml_parameters))

        elif xml_parameters.model_type == 'BayesianAtlas'.lower():
            assert args.command == 'estimate', \
                'The estimation of a bayesian atlas model should be launched with the command: ' \
                '"deformetrica estimate" (and not "%s").' % args.command
            deformetrica.estimate_bayesian_atlas(
                xml_parameters.template_specifications,
                get_dataset_specifications(xml_parameters),
                estimator_options=get_estimator_options(xml_parameters),
                model_options=get_model_options(xml_parameters))

        elif xml_parameters.model_type == 'PrincipalGeodesicAnalysis'.lower():
            assert args.command == 'estimate', \
                'The estimation of a principal geodesic analysis model should be launched with the command: ' \
                '"deformetrica estimate" (and not "%s").' % args.command
            deformetrica.estimate_principal_geodesic_analysis(
                xml_parameters.template_specifications,
                get_dataset_specifications(xml_parameters),
                estimator_options=get_estimator_options(xml_parameters),
                model_options=get_model_options(xml_parameters))

        elif xml_parameters.model_type == 'AffineAtlas'.lower():
            assert args.command == 'estimate', \
                'The estimation of a affine atlas model should be launched with the command: ' \
                '"deformetrica estimate" (and not "%s").' % args.command
            deformetrica.estimate_affine_atlas(
                xml_parameters.template_specifications,
                get_dataset_specifications(xml_parameters),
                estimator_options=get_estimator_options(xml_parameters),
                model_options=get_model_options(xml_parameters))

        elif xml_parameters.model_type == 'Regression'.lower():
            assert args.command == 'estimate', \
                'The estimation of a regression model should be launched with the command: ' \
                '"deformetrica estimate" (and not "%s").' % args.command
            deformetrica.estimate_geodesic_regression(
                xml_parameters.template_specifications,
                get_dataset_specifications(xml_parameters),
                estimator_options=get_estimator_options(xml_parameters),
                model_options=get_model_options(xml_parameters))

        elif xml_parameters.model_type == 'LongitudinalAtlas'.lower():
            assert args.command in ['estimate', 'initialize'], \
                'The initialization or estimation of a longitudinal atlas model should be launched with the command: ' \
                '"deformetrica {initialize, estimate}" (and not "%s").' % args.command
            if args.command == 'estimate':
                deformetrica.estimate_longitudinal_atlas(
                    xml_parameters.template_specifications,
                    get_dataset_specifications(xml_parameters),
                    estimator_options=get_estimator_options(xml_parameters),
                    model_options=get_model_options(xml_parameters))
            elif args.command == 'initialize':
                initialize_longitudinal_atlas(args.model, args.dataset, args.parameters, overwrite=True)

        elif xml_parameters.model_type == 'LongitudinalRegistration'.lower():
            assert args.command == 'estimate', \
                'The estimation of a longitudinal registration model should be launched with the command: ' \
                '"deformetrica estimate" (and not "%s").' % args.command
            deformetrica.estimate_longitudinal_registration(
                xml_parameters.template_specifications,
                get_dataset_specifications(xml_parameters),
                estimator_options=get_estimator_options(xml_parameters),
                model_options=get_model_options(xml_parameters))

        elif xml_parameters.model_type == 'Shooting'.lower():
            assert args.command == 'compute', \
                'The computation of a shooting task should be launched with the command: ' \
                '"deformetrica compute" (and not "%s").' % args.command
            deformetrica.compute_shooting(
                xml_parameters.template_specifications,
                model_options=get_model_options(xml_parameters))

        elif xml_parameters.model_type == 'ParallelTransport'.lower():
            assert args.command == 'compute', \
                'The computation of a parallel transport task should be launched with the command: ' \
                '"deformetrica compute" (and not "%s").' % args.command
            deformetrica.compute_parallel_transport(
                xml_parameters.template_specifications,
                model_options=get_model_options(xml_parameters))

        elif xml_parameters.model_type == 'LongitudinalMetricLearning'.lower():
            estimate_longitudinal_metric_model(xml_parameters)

        elif xml_parameters.model_type == 'LongitudinalMetricRegistration'.lower():
            estimate_longitudinal_metric_registration(xml_parameters)

        else:
            raise RuntimeError(
                'Unrecognized model-type: "' + xml_parameters.model_type + '". Check the corresponding field in the model.xml input file.')


if __name__ == "__main__":
    # execute only if run as a script
    main()
