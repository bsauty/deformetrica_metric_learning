import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

import numpy as np
import shutil
import math
from sklearn.decomposition import PCA
import torch
import xml.etree.ElementTree as et

from pydeformetrica.src.in_out.xml_parameters import XmlParameters
from pydeformetrica.src.launch.estimate_bayesian_atlas import estimate_bayesian_atlas
from pydeformetrica.src.launch.estimate_longitudinal_atlas import estimate_longitudinal_atlas
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
    Read command line, change directory, prepare preprocessing folder.
    """

    assert len(sys.argv) == 4, 'Usage: ' + sys.argv[0] + " <model.xml> <data_set.xml> <optimization_parameters.xml> "

    model_xml_path = sys.argv[1]
    dataset_xml_path = sys.argv[2]
    optimization_parameters_xml_path = sys.argv[3]

    preprocessings_folder = 'preprocessings'
    if not os.path.isdir(preprocessings_folder):
        os.mkdir(preprocessings_folder)

    """
    Compute an atlas on the baseline data.
    """

    # Clean folder.
    atlas_output_path = os.path.join(preprocessings_folder, '1_atlas_on_baseline_data')
    if os.path.isdir(atlas_output_path):
        shutil.rmtree(atlas_output_path)
        os.mkdir(atlas_output_path)

    # Read original longitudinal model xml parameters.
    xml_parameters = XmlParameters()
    xml_parameters._read_model_xml(model_xml_path)
    xml_parameters._read_dataset_xml(dataset_xml_path)
    xml_parameters._read_optimization_parameters_xml(optimization_parameters_xml_path)

    # Adapt the xml parameters and update.
    xml_parameters.model_type = 'BayesianAtlas'.lower()
    # xml_parameters.optimization_method_type = 'ScipyLBFGS'.lower()  # Works best in all cases.

    longitudinal_momenta = read_momenta(xml_parameters.initial_momenta).ravel()
    xml_parameters.initial_momenta = None

    xml_parameters.dataset_filenames = [[elt[0]] for elt in xml_parameters.dataset_filenames]
    xml_parameters.visit_ages = [[elt[0]] for elt in xml_parameters.visit_ages]

    xml_parameters._further_initialization()

    # Adapt the global settings, for the custom output directory.
    Settings().output_dir = atlas_output_path
    Settings().state_file = os.path.join(atlas_output_path, 'pydef_state.p')

    # Launch.
    estimate_bayesian_atlas(xml_parameters)

    """
    Tangent-space PCA on the momenta
    """

    # Load.
    control_points = read_2D_array(xml_parameters.initial_control_points)
    momenta = read_momenta(os.path.join(atlas_output_path, 'BayesianAtlas__Momenta.txt'))

    # Compute RKHS matrix.
    number_of_control_points = control_points.shape[0]
    dimension = Settings().dimension
    K = np.zeros((number_of_control_points * dimension, number_of_control_points * dimension))
    for i in range(number_of_control_points):
        for j in range(number_of_control_points):
            cp_i = control_points[i, :]
            cp_j = control_points[j, :]
            kernel_distance = math.exp(- np.sum((cp_j - cp_i) ** 2) / (xml_parameters.deformation_kernel_width ** 2))
            for d in range(dimension):
                K[dimension * i + d, dimension * j + d] = kernel_distance
                K[dimension * j + d, dimension * i + d] = kernel_distance

    # Project.
    kernel = create_kernel('exact', xml_parameters.deformation_kernel_width)

    Km = np.dot(K, longitudinal_momenta)
    mKm = np.dot(longitudinal_momenta.transpose(), Km)

    w = []
    for i in range(momenta.shape[0]):
        w.append(momenta[i].ravel() - np.dot(momenta[i].ravel(), Km) / mKm * longitudinal_momenta)

    # Pre-multiply.
    K_sqrt = np.linalg.cholesky(K)
    w = np.asarray([np.dot(K_sqrt, wi) for wi in w])

    # Standard PCA.
    if xml_parameters.number_of_sources is not None:
        number_of_sources = xml_parameters.number_of_sources
    elif xml_parameters.initial_modulation_matrix is not None:
        number_of_sources = read_2D_array(xml_parameters.initial_modulation_matrix).shape[1]
    else:
        number_of_sources = 4
        print('>> No initial modulation matrix given, neither a number of sources. '
              'The latter will be ARBITRARILY defaulted to 4.')

    pca = PCA(n_components=number_of_sources)
    pca.fit(w)

    # Save.
    initial_modulation_matrix = pca.components_.transpose()
    print('')
    print('[ tangent-space PCA ]')
    print('')
    print('>> ' + str(number_of_sources) + ' components: explained variance ratios = ')
    for s in range(number_of_sources):
        print(('\t %.3f %% \t[ Component ' + str(s) + ' ]') % (100.0 * pca.explained_variance_ratio_[s]))

    fn = 'ForInitialization_ModulationMatrix_FromAtlas.txt'
    write_2D_array(initial_modulation_matrix, fn)
    shutil.copyfile(os.path.join(atlas_output_path, fn), os.path.join('data', fn))

    # Modify the original model.xml file accordingly.
    found_tag = False
    model_xml_level0 = et.parse(model_xml_path).getroot()
    for model_xml_level1 in model_xml_level0:
        if model_xml_level1.tag.lower() == 'initial-modulation-matrix':
            model_xml_level1.text = os.path.join('data', fn)
            found_tag = True
    if not found_tag:
        initial_modulation_matrix_xml = et.SubElement(model_xml_level0, 'initial-modulation-matrix')
        initial_modulation_matrix_xml.text = os.path.join('data', fn)
    doc = et.ElementTree(model_xml_level0)
    model_xml_after_initialization_path = 'model_after_initialization.xml'
    doc.write(model_xml_after_initialization_path)

    """
    Registration of all target subjects.
    """

    print('')
    print('[ longitudinal registration of all subjects ]')
    print('')

    # Clean folder.
    registration_output_path = os.path.join(preprocessings_folder, '2_longitudinal_registration')
    if os.path.isdir(registration_output_path):
        shutil.rmtree(registration_output_path)
        os.mkdir(registration_output_path)

    # Read original longitudinal model xml parameters.
    xml_parameters = XmlParameters()
    xml_parameters._read_model_xml(model_xml_after_initialization_path)
    xml_parameters._read_dataset_xml(dataset_xml_path)
    xml_parameters._read_optimization_parameters_xml(optimization_parameters_xml_path)

    # Adapt the xml parameters and update.
    xml_parameters.model_type = 'LongitudinalRegistration'.lower()
    # xml_parameters.optimization_method_type = 'ScipyLBFGS'.lower()  # Works best in all cases.
    xml_parameters._further_initialization()

    # Adapt the global settings, for the custom output directory.
    Settings().output_dir = registration_output_path
    Settings().state_file = os.path.join(registration_output_path, 'pydef_state.p')

    # Launch.
    estimate_longitudinal_atlas(xml_parameters)


