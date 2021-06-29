import os
import sys
import time
import pandas as pd
from joblib import Parallel, delayed

sys.path.append('/home/benoit.sautydechalon/deformetrica')

import xml.etree.ElementTree as et

from deformetrica.in_out.xml_parameters import XmlParameters
from deformetrica.support.utilities.general_settings import Settings
from deformetrica import estimate_longitudinal_metric_model
from deformetrica.in_out.array_readers_and_writers import *
import xml.etree.ElementTree as et
from xml.dom.minidom import parseString
from deformetrica.in_out.dataset_functions import read_and_create_scalar_dataset, read_and_create_image_dataset
import deformetrica as dfca
from deformetrica.launch.estimate_longitudinal_metric_model import instantiate_longitudinal_metric_model
from deformetrica.core.estimators.gradient_ascent import GradientAscent
from deformetrica.in_out.dataset_functions import create_scalar_dataset



logger = logging.getLogger(__name__)
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logger.setLevel(logging.INFO)



dataset_used = 'simulated'
path = dataset_used + '_study/'

args = {'verbosity':'INFO', 'output':'personalize',
        'model':path+'model_personalize.xml', 'dataset':path+'data_set.xml', 'parameters':path+'optimization_parameters_saem.xml'}



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
#xml_parameters.optimization_method_type = 'ScipyLBFGS'.lower()

#xml_parameters.scale_initial_step_size = False
xml_parameters.max_iterations = 300
xml_parameters.max_line_search_iterations = 4

xml_parameters.initial_step_size = 0.1
xml_parameters.save_every_n_iters = 1000
xml_parameters.convergence_tolerance = 1e-4

# Freezing some variances !
xml_parameters.freeze_acceleration_variance = True
xml_parameters.freeze_noise_variance = True
xml_parameters.freeze_onset_age_variance = True

# Freezing other variables
xml_parameters.freeze_metric_parameters = True
xml_parameters.freeze_reference_time = True
xml_parameters.freeze_modulation_matrix = True
xml_parameters.freeze_p0 = True
xml_parameters.freeze_v0 = True
xml_parameters.output_dir = output_dir
Settings().output_dir = output_dir

logger.info(" >>> Personalizing for each individual.")

dataset = read_and_create_scalar_dataset(xml_parameters)
model, individual_RER = instantiate_longitudinal_metric_model(xml_parameters, logger, dataset,
                                                              observation_type=observation_type)
model.name = 'LongitudinalMetricLearning'

# Set the number of subject to 1 for the model to accept only one individual at a time
model.number_of_subjects = 1
individual_RER_sub = {}

datasets_individual_subjects = []
for i in range(dataset.number_of_subjects):

    for key in individual_RER.keys():
        individual_RER_sub[key] = np.array([individual_RER[key][i]])

    id_sub, data_sub, times_sub = dataset.subject_ids[i], dataset.deformable_objects[i], dataset.times[i]
    dataset_sub = create_scalar_dataset(np.array([id_sub for i in range(len(times_sub))]), np.array(data_sub.detach().numpy()), times_sub)

    datasets_individual_subjects.append((dataset_sub, individual_RER_sub))


def personalize_patient(dataset_sub, individual_RER_sub):
    Settings().dimension = xml_parameters.dimension
    estimator = GradientAscent(model, dataset_sub, 'GradientAscent', individual_RER_sub,
                               max_iterations=xml_parameters.max_iterations)
    estimator.initial_step_size = xml_parameters.initial_step_size
    estimator.max_line_search_iterations = xml_parameters.max_line_search_iterations
    estimator.optimized_log_likelihood = xml_parameters.optimized_log_likelihood
    estimator.convergence_tolerance = xml_parameters.convergence_tolerance
    estimator.update()
    return(estimator.individual_RER)

start_time = time.time()

#dataset_sub, individual_RER_sub = datasets_individual_subjects[46]
#test = personalize_patient(dataset_sub, individual_RER_sub)
#print(test)

individual_parameters = Parallel(n_jobs=6)(
            delayed(personalize_patient)(data_sub[0], data_sub[1]) for data_sub in datasets_individual_subjects)
          #delayed(personalize_patient)(i, datasets_individual_subjects[i][0], datasets_individual_subjects[i][1]) for i in range(len(datasets_individual_subjects)))

ind_params_df = pd.DataFrame(index=range(len(individual_parameters)), columns=['onset_age', 'log_acceleration', 'sources'])
for i in range(len(individual_parameters)):
    for feat in ind_params_df.columns:
        ind_params_df.loc[i][feat] = individual_parameters[i][feat][0]

ind_params_df.to_csv(path+'simulated_data_3/estimated_parameters.csv')

end_time = time.time()
logger.info(f">> Estimation took: {end_time - start_time}")


