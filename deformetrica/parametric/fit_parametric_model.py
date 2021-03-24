#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import argparse
import logging
import torch
import sys
from deformetrica.support.utilities.general_settings import Settings

sys.path.append('/Users/benoit.sautydechalon/deformetrica')
import deformetrica as dfca

logger = logging.getLogger(__name__)

args = {'command':'estimate', 'verbosity':'INFO', 'output':'output_test',
        'model':'model_after_initialization.xml', 'dataset':'data_set.xml', 'parameters':'optimization_parameters_saem.xml'}


 # set logging level
try:
    logger.setLevel(args['verbosity'])
except ValueError:
    logger.warning('Logging level was not recognized. Using INFO.')
    logger.setLevel(logging.INFO)

"""
Read xml files, set general settings, and call the adapted function.
"""

logger.info('Setting output directory to: ' + args['output'])
output_dir = args['output']

deformetrica = dfca.Deformetrica(output_dir=output_dir, verbosity=logger.level)

# logger.info('[ read_all_xmls function ]')
xml_parameters = dfca.io.XmlParameters()
xml_parameters.read_all_xmls(args['model'],
                             args['dataset'] if args['command'] == 'estimate' else None,
                             args['parameters'])

if xml_parameters.model_type == 'LongitudinalMetricLearning'.lower():
    dfca.estimate_longitudinal_metric_model(xml_parameters, logger)
else:
    raise RuntimeError(
        'Unrecognized model-type: "' + xml_parameters.model_type + '". Check the corresponding field in the model.xml input file.')


