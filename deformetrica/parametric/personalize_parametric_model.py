import os
import sys

sys.path.append('/Users/benoit.sautydechalon/deformetrica')

import xml.etree.ElementTree as et

from deformetrica.in_out.xml_parameters import XmlParameters
from deformetrica.support.utilities.general_settings import Settings
from deformetrica import estimate_longitudinal_metric_model
from deformetrica.in_out.array_readers_and_writers import *
import xml.etree.ElementTree as et
from xml.dom.minidom import parseString
from deformetrica.in_out.dataset_functions import read_and_create_scalar_dataset, read_and_create_image_dataset
import deformetrica as dfca


logger = logging.getLogger(__name__)
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logger.setLevel(logging.INFO)



dataset_used = 'bivariate'
path = dataset_used + '_study/'

args = {'verbosity':'INFO', 'output':'personalize',
        'model':path+'model_after_fit.xml', 'dataset':path+'data_set.xml', 'parameters':path+'optimization_parameters_saem.xml'}


"""
Read xml files, set general settings, and call the adapted function.
"""

logger.info('Setting output directory to: ' + args['output'])
output_dir = args['output']

deformetrica = dfca.Deformetrica(output_dir=output_dir, verbosity=logger.level)

# logger.info('[ read_all_xmls function ]')
xml_parameters = dfca.io.XmlParameters()
xml_parameters.read_all_xmls(args['model'],
                             args['dataset'],
                             args['parameters'])


# Creating the dataset object
dataset = read_and_create_scalar_dataset(xml_parameters)
observation_type = 'scalar'

"""
Gradient descent on the individual parameters 
"""

xml_parameters.optimization_method_type = 'GradientAscent'.lower()
xml_parameters.scale_initial_step_size = True
xml_parameters.max_iterations = 20
xml_parameters.save_every_n_iters = 1

# Freezing some variances !
xml_parameters.freeze_acceleration_variance = True
xml_parameters.freeze_metric_parameters = True
xml_parameters.freeze_noise_variance = True
xml_parameters.freeze_onset_age_variance = True
xml_parameters.freeze_reference_time = True

# Freezing other variables
xml_parameters.freeze_modulation_matrix = True
xml_parameters.freeze_p0 = True
xml_parameters.freeze_v0 = True
xml_parameters.output_dir = output_dir
Settings().output_dir = output_dir

logger.info(" >>> Performing gradient descent.")


estimate_longitudinal_metric_model(xml_parameters, logger=logger)

