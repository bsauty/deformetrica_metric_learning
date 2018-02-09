import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

import numpy as np
import torch
from torch.autograd import Variable
import warnings
import shutil
import math
from sklearn.decomposition import PCA
import torch
import xml.etree.ElementTree as et
from xml.dom.minidom import parseString

from pydeformetrica.src.in_out.xml_parameters import XmlParameters
from pydeformetrica.src.in_out.dataset_functions import create_template_metadata
from pydeformetrica.src.launch.estimate_bayesian_atlas import estimate_bayesian_atlas
from pydeformetrica.src.launch.estimate_deterministic_atlas import estimate_deterministic_atlas
from pydeformetrica.src.launch.estimate_geodesic_regression import estimate_geodesic_regression
from pydeformetrica.src.core.model_tools.deformations.geodesic import Geodesic
from pydeformetrica.src.launch.estimate_longitudinal_registration import estimate_longitudinal_registration
from pydeformetrica.src.support.utilities.general_settings import Settings
from src.in_out.utils import *
from pydeformetrica.src.support.kernels.kernel_functions import create_kernel


def insert_model_xml_level1_entry(model_xml_level0, key, value):
    found_tag = False
    for model_xml_level1 in model_xml_level0:
        if model_xml_level1.tag.lower() == key:
            model_xml_level1.text = value
            found_tag = True
    if not found_tag:
        new_element_xml = et.SubElement(model_xml_level0, key)
        new_element_xml.text = value
    return model_xml_level0


def insert_model_xml_template_spec_entry(model_xml_level0, key, values):
    for model_xml_level1 in model_xml_level0:
        if model_xml_level1.tag.lower() == 'template':
            for k, model_xml_level2 in enumerate(model_xml_level1):
                found_tag = False
                for model_xml_level3 in model_xml_level2:
                    if model_xml_level3.tag.lower() == key.lower():
                        model_xml_level3.text = values[k]
                        found_tag = True
                if not found_tag:
                    new_element_xml = et.SubElement(model_xml_level2, key)
                    new_element_xml.text = values[k]
    return model_xml_level0


if __name__ == '__main__':

    print('')
    print('##############################')
    print('##### PyDeformetrica 1.0 #####')
    print('##############################')
    print('')

    """
    0]. Read command line, change directory, prepare preprocessing folder.
    """

    assert len(sys.argv) == 4, 'Usage: ' + sys.argv[0] + " <model.xml> <data_set.xml> <optimization_parameters.xml> "

    model_xml_path = sys.argv[1]
    dataset_xml_path = sys.argv[2]
    optimization_parameters_xml_path = sys.argv[3]

    preprocessings_folder = 'preprocessings'
    if not os.path.isdir(preprocessings_folder):
        os.mkdir(preprocessings_folder)

    """
    1]. Compute an atlas on the baseline data.
    ------------------------------------------
        The outputted template, control points and noise standard deviation will be used in the following
        geodesic regression and longitudinal registration, as well as an initialization for the longitudinal atlas.
        The template will always be used, i.e. the user-provided one is assumed to be a dummy, low-quality one.
        On the other hand, the estimated control points and noise standard deviation will only be used if the user did
        not provide those.
    """

    print('')
    print('[ estimate an atlas from baseline data ]')
    print('')

    # Initialization ---------------------------------------------------------------------------------------------------
    # Clean folder.
    atlas_output_path = os.path.join(preprocessings_folder, '1_atlas_on_baseline_data')
    if os.path.isdir(atlas_output_path): shutil.rmtree(atlas_output_path)
    os.mkdir(atlas_output_path)

    # Read original longitudinal model xml parameters.
    xml_parameters = XmlParameters()
    xml_parameters._read_model_xml(model_xml_path)
    xml_parameters._read_dataset_xml(dataset_xml_path)
    xml_parameters._read_optimization_parameters_xml(optimization_parameters_xml_path)

    # Save some global parameters.
    global_full_dataset_filenames = xml_parameters.dataset_filenames
    global_full_visit_ages = xml_parameters.visit_ages
    global_full_subject_ids = xml_parameters.subject_ids

    global_number_of_subjects = len(global_full_dataset_filenames)
    global_number_of_timepoints = sum([len(elt) for elt in global_full_visit_ages])

    global_initial_control_points_are_given = xml_parameters.initial_control_points is not None
    global_initial_momenta_are_given = xml_parameters.initial_momenta is not None
    global_initial_modulation_matrix_is_given = xml_parameters.initial_modulation_matrix is not None
    global_initial_t0_is_given = xml_parameters.t0 is not None

    global_t0 = xml_parameters.t0
    if not global_initial_t0_is_given:
        global_t0 = sum([sum(elt) for elt in global_full_visit_ages]) / float(global_number_of_timepoints)

    global_tmin = sum([elt[0] for elt in global_full_visit_ages]) / float(global_number_of_subjects)

    # Adapt the xml parameters and update.
    # atlas_type = 'Bayesian'
    atlas_type = 'Deterministic'
    xml_parameters.model_type = (atlas_type + 'Atlas').lower()
    xml_parameters.optimization_method_type = 'ScipyLBFGS'.lower()

    xml_parameters.initial_momenta = None

    xml_parameters.dataset_filenames = [[elt[0]] for elt in xml_parameters.dataset_filenames]
    xml_parameters.visit_ages = [[elt[0]] for elt in xml_parameters.visit_ages]

    xml_parameters._further_initialization()

    # Adapt the global settings, for the custom output directory.
    Settings().output_dir = atlas_output_path
    Settings().state_file = os.path.join(atlas_output_path, 'pydef_state.p')

    # Launch and save the outputted noise standard deviation, for later use --------------------------------------------
    if atlas_type == 'Bayesian':
        model = estimate_bayesian_atlas(xml_parameters)
        global_objects_noise_std = read_2D_array(os.path.join(atlas_output_path,
                                                              'BayesianAtlas__EstimatedParameters__NoiseStd.txt'))

    elif atlas_type == 'Deterministic':
        model = estimate_deterministic_atlas(xml_parameters)
        global_objects_noise_std = [math.sqrt(elt) for elt in model.objects_noise_variance]

    else:
        raise RuntimeError('Unknown atlas type: "' + atlas_type + '"')

    # Export the results -----------------------------------------------------------------------------------------------
    global_objects_name, global_objects_name_extension, original_objects_noise_variance = \
        create_template_metadata(xml_parameters.template_specifications)[1:4]

    global_initial_template = model.template
    global_initial_template_data = model.get_template_data()
    global_initial_control_points = model.get_control_points()
    global_initial_objects_template_path = []
    for k, (object_name, object_name_extension, original_object_noise_variance) \
            in enumerate(zip(global_objects_name, global_objects_name_extension, original_objects_noise_variance)):

        # Copy the estimated template to the data folder.
        estimated_template_path = os.path.join(
            atlas_output_path,
            atlas_type + 'Atlas__EstimatedParameters__Template_' + object_name + object_name_extension)
        global_initial_objects_template_path.append(os.path.join(
            'data', 'ForInitialization__Template_' + object_name + '__FromAtlas' + object_name_extension))
        shutil.copyfile(estimated_template_path, global_initial_objects_template_path[k])

        if Settings().dimension == 2:
            cmd_replace = 'sed -i -- s/POLYGONS/LINES/g ' + global_initial_objects_template_path[k]
            os.system(cmd_replace)  # Quite time-consuming.
            if os.path.isfile(global_initial_objects_template_path[k] + '--'):
                os.remove(global_initial_objects_template_path[k] + '--')

        # Override the obtained noise standard deviation values, if it was already given by the user.
        if original_object_noise_variance > 0: global_objects_noise_std[k] = math.sqrt(original_object_noise_variance)

    # Convert the noise std float values to formatted strings.
    global_objects_noise_std_string = ['{:.4f}'.format(elt) for elt in global_objects_noise_std]

    # If necessary, copy the estimated control points to the data folder.
    if global_initial_control_points_are_given:
        global_initial_control_points_path = xml_parameters.initial_control_points
        global_initial_control_points = read_2D_array(global_initial_control_points_path)
    else:
        estimated_control_points_path = os.path.join(atlas_output_path,
                                                     atlas_type + 'Atlas__EstimatedParameters__ControlPoints.txt')
        global_initial_control_points_path = os.path.join('data', 'ForInitialization__ControlPoints__FromAtlas.txt')
        shutil.copyfile(estimated_control_points_path, global_initial_control_points_path)

    # Modify and write the model.xml file accordingly.
    model_xml_level0 = et.parse(model_xml_path).getroot()
    model_xml_level0 = insert_model_xml_template_spec_entry(model_xml_level0,
                                                            'filename', global_initial_objects_template_path)
    model_xml_level0 = insert_model_xml_template_spec_entry(model_xml_level0,
                                                            'noise-std', global_objects_noise_std_string)
    model_xml_level0 = insert_model_xml_level1_entry(model_xml_level0,
                                                     'initial-control-points', global_initial_control_points_path)
    model_xml_path = 'initialized_model.xml'
    doc = parseString((et.tostring(model_xml_level0).decode('utf-8').replace('\n', '').replace('\t', ''))).toprettyxml()
    np.savetxt(model_xml_path, [doc], fmt='%s')

    """
    2]. Compute individual geodesic regressions.
    --------------------------------------------
        The time t0 is chosen as the baseline age for every subject.
        The control points are the one outputted by the atlas estimation, and are frozen.
        Skipped if an initial control points and (longitudinal) momenta are specified.
    """

    # Read the current model xml parameters.
    xml_parameters = XmlParameters()
    xml_parameters._read_model_xml(model_xml_path)
    xml_parameters._read_dataset_xml(dataset_xml_path)
    xml_parameters._read_optimization_parameters_xml(optimization_parameters_xml_path)

    # Check if an initial (longitudinal) momenta is available.
    if global_initial_control_points_are_given and global_initial_momenta_are_given:
        global_initial_momenta = read_3D_array(xml_parameters.initial_momenta)

    else:
        print('')
        print('[ compute individual geodesic regressions ]')

        # Warning.
        if not global_initial_control_points_are_given and global_initial_momenta_are_given:
            msg = 'Initial momenta are given but not the corresponding initial control points. ' \
                  'Those given initial momenta will be ignored, and overridden by a regression-based heuristic.'
            warnings.warn(msg)

        # Create folder.
        regressions_output_path = os.path.join(preprocessings_folder, '2_individual_geodesic_regressions')
        if os.path.isdir(regressions_output_path): shutil.rmtree(regressions_output_path)
        os.mkdir(regressions_output_path)

        # Adapt the shared xml parameters.
        xml_parameters.model_type = 'Regression'.lower()
        xml_parameters.optimization_method_type = 'ScipyLBFGS'.lower()
        xml_parameters.freeze_control_points = True

        # Loop over each subject.
        global_initial_momenta = np.zeros(read_2D_array(xml_parameters.initial_control_points).shape)
        for i in range(global_number_of_subjects):

            print('')
            print('[ geodesic regression for subject ' + global_full_subject_ids[i] + ' ]')
            print('')

            # Create folder.
            subject_regression_output_path = os.path.join(regressions_output_path,
                                                          'GeodesicRegression__subject_' + global_full_subject_ids[i])
            if os.path.isdir(subject_regression_output_path): shutil.rmtree(subject_regression_output_path)
            os.mkdir(subject_regression_output_path)

            # Adapt the specific xml parameters and update.
            xml_parameters.dataset_filenames = [global_full_dataset_filenames[i]]
            xml_parameters.visit_ages = [global_full_visit_ages[i]]
            xml_parameters.subject_ids = [global_full_subject_ids[i]]
            xml_parameters.t0 = xml_parameters.visit_ages[0][0]
            xml_parameters._further_initialization()

            # Adapt the global settings, for the custom output directory.
            Settings().output_dir = subject_regression_output_path
            Settings().state_file = os.path.join(subject_regression_output_path, 'pydef_state.p')

            # Launch.
            estimate_geodesic_regression(xml_parameters)

            # Add the estimated momenta.
            global_initial_momenta += read_3D_array(os.path.join(
                subject_regression_output_path, 'GeodesicRegression__EstimatedParameters__Momenta.txt'))

        # Divide to obtain the average momenta. Write the result in the data folder.
        global_initial_momenta /= float(global_number_of_subjects)
        global_initial_momenta_path = os.path.join('data', 'ForInitialization__Momenta__FromRegressions.txt')
        np.savetxt(global_initial_momenta_path, global_initial_momenta)

        # Modify and write the model.xml file accordingly.
        model_xml_level0 = et.parse(model_xml_path).getroot()
        model_xml_level0 = insert_model_xml_level1_entry(model_xml_level0,
                                                         'initial-momenta', global_initial_momenta_path)
        model_xml_path = 'initialized_model.xml'
        doc = parseString(
            (et.tostring(model_xml_level0).decode('utf-8').replace('\n', '').replace('\t', ''))).toprettyxml()
        np.savetxt(model_xml_path, [doc], fmt='%s')

    """
    3]. Shoot from the average baseline age to the global average.
    --------------------------------------------------------------
        New values are obtained for the template, control points, and (longitudinal) momenta.
        Skipped if initial control points and momenta were given.
    """

    if not global_initial_control_points_are_given and not global_initial_momenta_are_given:

        print('')
        print('[ shoot from the average baseline age to the global average ]')
        print('')

        # Shoot --------------------------------------------------------------------------------------------------------
        # Create folder.
        shooting_output_path = os.path.join(preprocessings_folder, '3_shooting_from_baseline_to_average')
        if os.path.isdir(shooting_output_path): shutil.rmtree(shooting_output_path)
        os.mkdir(shooting_output_path)

        # Instantiate a geodesic.
        geodesic = Geodesic()
        geodesic.set_kernel(create_kernel(xml_parameters.deformation_kernel_type,
                                          xml_parameters.deformation_kernel_width))
        geodesic.concentration_of_time_points = xml_parameters.concentration_of_time_points
        geodesic.set_use_rk2(xml_parameters.use_rk2)
        geodesic.set_t0(xml_parameters.t0)
        geodesic.set_tmin(global_tmin)
        geodesic.set_tmax(global_t0)

        # Set the template, control points and momenta and update.
        geodesic.set_template_data_t0(Variable(torch.from_numpy(
            global_initial_template_data).type(Settings().tensor_scalar_type), requires_grad=False))
        geodesic.set_control_points_t0(Variable(torch.from_numpy(
            global_initial_control_points).type(Settings().tensor_scalar_type), requires_grad=False))
        geodesic.set_momenta_t0(Variable(torch.from_numpy(
            global_initial_momenta).type(Settings().tensor_scalar_type), requires_grad=False))
        geodesic.update()

        # Adapt the global settings, for the custom output directory.
        Settings().output_dir = shooting_output_path

        # Write.
        geodesic.write('Shooting', global_objects_name, global_objects_name_extension,
                       global_initial_template, write_shoot=True)

        # Export results -----------------------------------------------------------------------------------------------
        number_of_timepoints = \
            geodesic.forward_exponential.number_of_time_points + geodesic.backward_exponential.number_of_time_points - 2

        # Template.
        for k, (object_name, object_name_extension) in enumerate(
                zip(global_objects_name, global_objects_name_extension)):
            # Copy the estimated template to the data folder.
            shooted_template_path = os.path.join(
                shooting_output_path, 'Shooting__GeodesicFlow__' + object_name + '__tp_' + str(number_of_timepoints)
                                      + ('__age_%.2f' % global_t0) + object_name_extension)
            global_initial_objects_template_path[k] = os.path.join(
                'data', 'ForInitialization__Template_' + object_name + '__FromShooting' + object_name_extension)
            shutil.copyfile(shooted_template_path, global_initial_objects_template_path[k])

            if Settings().dimension == 2:
                cmd_replace = 'sed -i -- s/POLYGONS/LINES/g ' + global_initial_objects_template_path[k]
                os.system(cmd_replace)  # Quite time-consuming.
                if os.path.isfile(global_initial_objects_template_path[k] + '--'):
                    os.remove(global_initial_objects_template_path[k] + '--')

        # Control points.
        shooted_control_points_path = os.path.join(
            shooting_output_path, 'Shooting__GeodesicFlow__ControlPoints__tp_' + str(number_of_timepoints)
                                  + ('__age_%.2f' % global_t0) + '.txt')
        global_initial_control_points_path = os.path.join('data', 'ForInitialization__ControlPoints__FromShooting.txt')
        shutil.copyfile(shooted_control_points_path, global_initial_control_points_path)

        # Momenta.
        shooted_momenta_path = os.path.join(
            shooting_output_path, 'Shooting__GeodesicFlow__Momenta__tp_' + str(number_of_timepoints)
                                  + ('__age_%.2f' % global_t0) + '.txt')
        global_initial_momenta_path = os.path.join('data', 'ForInitialization__Momenta__FromShooting.txt')
        shutil.copyfile(shooted_momenta_path, global_initial_momenta_path)
        global_initial_momenta = read_3D_array(global_initial_momenta_path)

        # Modify and write the model.xml file accordingly.
        model_xml_level0 = et.parse(model_xml_path).getroot()
        model_xml_level0 = insert_model_xml_template_spec_entry(model_xml_level0,
                                                                'filename', global_initial_objects_template_path)
        model_xml_level0 = insert_model_xml_level1_entry(model_xml_level0,
                                                         'initial-control-points', global_initial_control_points_path)
        model_xml_level0 = insert_model_xml_level1_entry(model_xml_level0,
                                                         'initial-momenta', global_initial_momenta_path)

        model_xml_path = 'initialized_model.xml'
        doc = parseString(
            (et.tostring(model_xml_level0).decode('utf-8').replace('\n', '').replace('\t', ''))).toprettyxml()
        np.savetxt(model_xml_path, [doc], fmt='%s')

    """
    4]. Tangent-space PCA on the individual momenta outputted by the atlas estimation.
    ----------------------------------------------------------------------------------
        Those momenta are first projected on the space orthogonal to the initial (longitudinal) momenta.
        Skipped if initial control points and modulation matrix were specified.
    """

    # Check if an initial (longitudinal) momenta is available.
    if not (global_initial_control_points_are_given and global_initial_modulation_matrix_is_given):

        print('')
        print('[ tangent-space PCA on the projected individual momenta ]')
        print('')

        # Warning.
        if not global_initial_control_points_are_given and global_initial_modulation_matrix_is_given:
            msg = 'Initial modulation matrix is given but not the corresponding initial control points. ' \
                  'This given initial modulation matrix will be ignored, and overridden by a PCA-based heuristic.'
            warnings.warn(msg)

        # Read the current model xml parameters.
        xml_parameters = XmlParameters()
        xml_parameters._read_model_xml(model_xml_path)
        xml_parameters._read_dataset_xml(dataset_xml_path)
        xml_parameters._read_optimization_parameters_xml(optimization_parameters_xml_path)

        # Load.
        control_points = read_2D_array(xml_parameters.initial_control_points)
        momenta = read_3D_array(os.path.join(atlas_output_path, atlas_type + 'Atlas__EstimatedParameters__Momenta.txt'))

        # Compute RKHS matrix.
        number_of_control_points = control_points.shape[0]
        dimension = Settings().dimension
        K = np.zeros((number_of_control_points * dimension, number_of_control_points * dimension))
        for i in range(number_of_control_points):
            for j in range(number_of_control_points):
                cp_i = control_points[i, :]
                cp_j = control_points[j, :]
                kernel_distance = math.exp(
                    - np.sum((cp_j - cp_i) ** 2) / (xml_parameters.deformation_kernel_width ** 2))
                for d in range(dimension):
                    K[dimension * i + d, dimension * j + d] = kernel_distance
                    K[dimension * j + d, dimension * i + d] = kernel_distance

        # Project.
        kernel = create_kernel('exact', xml_parameters.deformation_kernel_width)

        Km = np.dot(K, global_initial_momenta.ravel())
        mKm = np.dot(global_initial_momenta.ravel().transpose(), Km)

        w = []
        for i in range(momenta.shape[0]):
            w.append(momenta[i].ravel() - np.dot(momenta[i].ravel(), Km) / mKm * global_initial_momenta.ravel())

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
        global_initial_modulation_matrix = pca.components_.transpose()
        print('>> ' + str(number_of_sources) + ' components: explained variance ratios = ')
        for s in range(number_of_sources):
            print(('\t %.3f %% \t[ Component ' + str(s) + ' ]') % (100.0 * pca.explained_variance_ratio_[s]))

        global_initial_modulation_matrix_path = \
            os.path.join('data', 'ForInitialization__ModulationMatrix__FromPCA.txt')
        np.savetxt(global_initial_modulation_matrix_path, global_initial_modulation_matrix)

        # Modify the original model.xml file accordingly.
        model_xml_level0 = et.parse(model_xml_path).getroot()
        model_xml_level0 = insert_model_xml_level1_entry(
            model_xml_level0, 'initial-modulation-matrix', global_initial_modulation_matrix_path)
        model_xml_path = 'initialized_model.xml'
        doc = parseString(
            (et.tostring(model_xml_level0).decode('utf-8').replace('\n', '').replace('\t', ''))).toprettyxml()
        np.savetxt(model_xml_path, [doc], fmt='%s')

    """
    5]. Longitudinal registration of all target subjects.
    -----------------------------------------------------
        The reference is the average of the ages at all visits.
        The template, control points and modulation matrix are from the atlas estimation.
        The momenta is from the individual regressions.
    """

    print('')
    print('[ longitudinal registration of all subjects ]')
    print('')

    # Clean folder.
    registration_output_path = os.path.join(preprocessings_folder, '4_longitudinal_registration')
    if os.path.isdir(registration_output_path): shutil.rmtree(registration_output_path)
    os.mkdir(registration_output_path)

    # Read original longitudinal model xml parameters.
    xml_parameters = XmlParameters()
    xml_parameters._read_model_xml(model_xml_path)
    xml_parameters._read_dataset_xml(dataset_xml_path)
    xml_parameters._read_optimization_parameters_xml(optimization_parameters_xml_path)

    # Adapt the xml parameters and update.
    xml_parameters.model_type = 'LongitudinalRegistration'.lower()
    xml_parameters.optimization_method_type = 'ScipyPowell'.lower()
    xml_parameters._further_initialization()

    # Adapt the global settings, for the custom output directory.
    Settings().output_dir = registration_output_path

    # Launch.
    estimate_longitudinal_registration(xml_parameters)

    # Copy the output individual effects into the data folder.
    estimated_onset_ages_path = os.path.join(registration_output_path,
                                             'LongitudinalRegistration__EstimatedParameters__OnsetAges.txt')
    initial_onset_ages_path = os.path.join('data', 'ForInitialization_OnsetAges_FromLongitudinalRegistration.txt')
    shutil.copyfile(estimated_onset_ages_path, initial_onset_ages_path)

    estimated_log_accelerations_path = os.path.join(
        registration_output_path, 'LongitudinalRegistration__EstimatedParameters__LogAccelerations.txt')
    initial_log_accelerations_path = os.path.join(
        'data', 'ForInitialization__LogAccelerations__FromLongitudinalRegistration.txt')
    shutil.copyfile(estimated_log_accelerations_path, initial_log_accelerations_path)

    estimated_sources_path = os.path.join(registration_output_path,
                                          'LongitudinalRegistration__EstimatedParameters__Sources.txt')
    initial_sources_path = os.path.join('data', 'ForInitialization__Sources__FromLongitudinalRegistration.txt')
    shutil.copyfile(estimated_sources_path, initial_sources_path)

    # Modify the original model.xml file accordingly.
    model_xml_level0 = et.parse(model_xml_path).getroot()
    model_xml_level0 = insert_model_xml_level1_entry(model_xml_level0, 'initial-onset-ages', initial_onset_ages_path)
    model_xml_level0 = insert_model_xml_level1_entry(model_xml_level0,
                                                     'initial-log-accelerations', initial_log_accelerations_path)
    model_xml_level0 = insert_model_xml_level1_entry(model_xml_level0, 'initial-sources', initial_sources_path)
    model_xml_path = 'initialized_model.xml'
    doc = parseString((et.tostring(model_xml_level0).decode('utf-8').replace('\n', '').replace('\t', ''))).toprettyxml()
    np.savetxt(model_xml_path, [doc], fmt='%s')
