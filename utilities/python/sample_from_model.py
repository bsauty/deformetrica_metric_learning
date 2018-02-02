import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

import numpy as np
import shutil
import math
from sklearn.decomposition import PCA
import torch
import xml.etree.ElementTree as et
from xml.dom.minidom import parseString
from numpy.random import poisson, exponential, normal

from pydeformetrica.src.in_out.xml_parameters import XmlParameters
from pydeformetrica.src.core.observations.datasets.longitudinal_dataset import LongitudinalDataset
from pydeformetrica.src.core.models.longitudinal_atlas import LongitudinalAtlas
from pydeformetrica.src.launch.estimate_bayesian_atlas import estimate_bayesian_atlas
from pydeformetrica.src.launch.estimate_longitudinal_atlas import instantiate_longitudinal_atlas_model
from pydeformetrica.src.support.utilities.general_settings import Settings
from src.in_out.utils import *
from pydeformetrica.src.support.kernels.kernel_functions import create_kernel

if __name__ == '__main__':

    """
    Basic info printing.
    """

    print('')
    print('##############################')
    print('##### PyDeformetrica 1.0 #####')
    print('##############################')
    print('')

    """
    Read command line, create output directory, read the model xml file.
    """

    assert len(sys.argv) == 5, \
        'Usage: ' + sys.argv[0] + " <model.xml> <number_of_subjects> " \
                                  "<mean_number_of_visits_minus_two> " "<mean_observation_time_window>"

    model_xml_path = sys.argv[1]
    number_of_subjects = int(sys.argv[2])
    mean_number_of_visits_minus_two = float(sys.argv[3])
    mean_observation_time_window = float(sys.argv[4])

    k = 1
    sample_folder = 'sample_' + str(k)
    while os.path.isdir(sample_folder):
        k += 1
        sample_folder = 'sample_' + str(k)
    os.mkdir(sample_folder)

    xml_parameters = XmlParameters()
    xml_parameters._read_model_xml(model_xml_path)

    if xml_parameters.model_type == 'LongitudinalAtlas'.lower():

        """
        Instantiate the model.
        """
        model, _ = instantiate_longitudinal_atlas_model(xml_parameters, ignore_noise_variance=True)

        """
        Draw random visit ages and create a degenerated dataset object.
        """

        visit_ages = []
        for i in range(number_of_subjects):
            number_of_visits = 2 + poisson(mean_number_of_visits_minus_two)
            observation_time_window = exponential(mean_observation_time_window)

            time_between_two_consecutive_visits = observation_time_window / float(number_of_visits - 1)
            age_at_baseline = normal(model.get_reference_time(), math.sqrt(model.get_time_shift_variance())) \
                              - 0.5 * observation_time_window

            ages = [age_at_baseline + j * time_between_two_consecutive_visits for j in range(number_of_visits)]
            visit_ages.append(ages)

        dataset = LongitudinalDataset()
        dataset.times = visit_ages
        dataset.subject_ids = ['s' + str(i) for i in range(number_of_subjects)]
        dataset.number_of_subjects = number_of_subjects
        dataset.total_number_of_observations = sum([len(elt) for elt in visit_ages])

        print('>> ' + str(number_of_subjects) + ' will be generated, with '
              + str(float(dataset.total_number_of_observations) / float(number_of_subjects)) + ' visits on average.')

        """
        Generate individual RER.
        """

        onset_ages = np.zeros((number_of_subjects,))
        log_accelerations = np.zeros((number_of_subjects,))
        sources = np.zeros((number_of_subjects, model.number_of_sources))

        for i in range(number_of_subjects):
            onset_ages[i] = model.individual_random_effects['onset_age'].sample()
            log_accelerations[i] = model.individual_random_effects['log_acceleration'].sample()
            sources[i] = model.individual_random_effects['sources'].sample()

        individual_RER = {}
        individual_RER['sources'] = sources
        individual_RER['onset_age'] = onset_ages
        individual_RER['log_acceleration'] = log_accelerations

        """
        Call the write method of the model.
        """
        model.write(dataset, None, individual_RER, update_fixed_effects=False)

    else:
        msg = 'Sampling from the specified "' + xml_parameters.model_type + '" model is not available yet.'
        raise RuntimeError(msg)
