import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

import numpy as np
import shutil

from pydeformetrica.src.in_out.xml_parameters import XmlParameters
from pydeformetrica.src.in_out.dataset_functions import create_dataset
from pydeformetrica.src.launch.estimate_bayesian_atlas import estimate_bayesian_atlas
from pydeformetrica.src.support.utilities.general_settings import Settings



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
    Read command line, change directory, prepare preprocessing folder.
    """

    assert len(sys.argv) == 2, 'Usage: ' + sys.argv[0] + ' path/to/longitudinal/folder'
    longitudinal_folder_path = sys.argv[1]

    os.chdir(longitudinal_folder_path)

    preprocessings_folder = 'preprocessings'
    if not os.path.isdir(preprocessings_folder):
        os.mkdir(preprocessings_folder)

    """
    Compute an atlas on the baseline data.
    """

    # Clean folder.
    atlas_output_path = os.path.join(preprocessings_folder, 'atlas_on_baseline_data')
    if os.path.isdir(atlas_output_path):
        shutil.rmtree(atlas_output_path)
        os.mkdir(atlas_output_path)

    # Read original longitudinal model xml parameters.
    model_xml_path = os.path.join(longitudinal_folder_path, 'model.xml')
    dataset_xml_path = os.path.join(longitudinal_folder_path, 'data_set.xml')
    optimization_parameters_xml_path = os.path.join(longitudinal_folder_path, 'optimization_parameters.xml')

    xml_parameters = XmlParameters()
    xml_parameters._read_model_xml(model_xml_path)
    xml_parameters._read_dataset_xml(dataset_xml_path)
    xml_parameters._read_optimization_parameters_xml(optimization_parameters_xml_path)

    # Adapt the xml parameters and update.
    xml_parameters.model_type = 'BayesianAtlas'.lower()
    xml_parameters.optimization_method_type = 'ScipyLBFGS'.lower()  # Works best in all cases.
    xml_parameters.initial_momenta = None

    xml_parameters.dataset_filenames = [[elt[0]] for elt in xml_parameters.dataset_filenames]
    xml_parameters.visit_ages = [[elt[0]] for elt in xml_parameters.visit_ages]

    xml_parameters._further_initialization()

    # Adapt the global settings, for the custom output directory.
    Settings().output_dir = atlas_output_path

    # Launch.
    estimate_bayesian_atlas(xml_parameters)






