#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import argparse
import logging
import torch
import sys

sys.path.append('/home/benoit.sautydechalon/deformetrica')
import deformetrica as dfca
from deformetrica.support.utilities.general_settings import Settings


logger = logging.getLogger(__name__)

dataset_used = 'simulated'
path = dataset_used + '_study/'


args = {'command':'estimate', 'verbosity':'INFO', 'output':'output_7',
        'model':path+'model_after_initialization_7.xml', 'dataset':path+'data_set.xml', 'parameters':path+'optimization_parameters_saem.xml'}

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
Settings().output_dir = output_dir

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


