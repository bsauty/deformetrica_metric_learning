#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import argparse

from in_out.xml_parameters import XmlParameters
from launch.compute_parallel_transport import compute_parallel_transport
from launch.estimate_bayesian_atlas import estimate_bayesian_atlas
from launch.estimate_deterministic_atlas import estimate_deterministic_atlas
from launch.estimate_geodesic_regression import estimate_geodesic_regression
from launch.estimate_longitudinal_atlas import estimate_longitudinal_atlas
from launch.estimate_longitudinal_metric_model import estimate_longitudinal_metric_model
from launch.estimate_longitudinal_metric_registration import estimate_longitudinal_metric_registration
from launch.estimate_longitudinal_registration import estimate_longitudinal_registration
from launch.run_shooting import run_shooting
from support.utilities.general_settings import Settings


def info():
    version = open(os.path.dirname(os.path.realpath(__file__)) + '/../VERSION').read()
    return """
    ##############################
    ##### Deformetrica {version} #####
    ##############################
    """.format(version=version)


def main():
    import logging
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)

    # parse arguments
    parser = argparse.ArgumentParser(description='Deformetrica')
    parser.add_argument('model', type=str, help='model xml file')
    parser.add_argument('dataset', type=str, help='data-set xml file')
    parser.add_argument('optimization', type=str, help='optimization parameters xml file')

    # optional arguments
    parser.add_argument('-o', '--output', type=str, help='output folder')
    parser.add_argument('-v', '--verbosity', action='count', default=0, help='increase output verbosity')  # TODO

    args = parser.parse_args()

    # Basic info printing
    print(info())

    """
    Read xml files, set general settings, and call the adapted function.
    """
    try:
        if args.output is None:
            logger.info('Creating the output directory: ' + Settings().output_dir)
            os.makedirs(Settings().output_dir)
        else:
            logger.info('Setting output directory to: ' + args.output)
            Settings().set_output_dir(args.output)
    except FileExistsError:
        pass

    logger.info('[ read_all_xmls function ]')
    xml_parameters = XmlParameters()
    xml_parameters.read_all_xmls(args.model, args.dataset, args.optimization)

    if xml_parameters.model_type == 'DeterministicAtlas'.lower() \
            or xml_parameters.model_type == 'Registration'.lower():
        estimate_deterministic_atlas(xml_parameters)

    elif xml_parameters.model_type == 'BayesianAtlas'.lower():
        estimate_bayesian_atlas(xml_parameters)

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


if __name__ == "__main__":
    # execute only if run as a script
    main()