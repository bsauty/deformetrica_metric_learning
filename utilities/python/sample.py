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

from pydeformetrica.src.in_out.xml_parameters import XmlParameters
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

    assert len(sys.argv) == 5, 'Usage: ' + sys.argv[0] + " <model.xml> <number_of_subjects> <mean_number_of_visits> " \
                                                         "<mean_observation_time_window>"

    model_xml_path = sys.argv[1]
    number_of_subjects = int(sys.argv[2])
    mean_number_of_visits = float(sys.argv[3])
    mean_observation_time_window = float(sys.argv[4])

    k = 1
    sample_folder = 'sample_' + str(k)
    while os.path.isdir(sample_folder):
        k += 1
        sample_folder = 'sample_' + str(k)
    os.mkdir(sample_folder)

    xml_parameters = XmlParameters()
    xml_parameters._read_model_xml(model_xml_path)

    """
    Instantiate the model.
    """

    if xml_parameters.model_type == 'LongitudinalAtlas'.lower():
        model = instantiate_longitudinal_atlas_model(xml_parameters, ignore_noise_variance=True)

        # Create dataset object, possibly degenerated (only visit ages)
        # Generate individual RER
        # Call write method of the longitudinal atlas, without computation of the residuals + update


    else:
        msg = 'Sampling from the specified "' + xml_parameters.model_type + '" model is not available yet.'
        raise RuntimeError(msg)