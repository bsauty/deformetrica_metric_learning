import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

import numpy as np
import torch
from torch.autograd import Variable
from torch.multiprocessing import Pool

import warnings
from decimal import Decimal
import shutil
import math
from sklearn.decomposition import PCA, FastICA
import torch
import xml.etree.ElementTree as et
from xml.dom.minidom import parseString

from pydeformetrica.src.in_out.xml_parameters import XmlParameters
from pydeformetrica.src.in_out.dataset_functions import create_template_metadata
from pydeformetrica.src.launch.estimate_bayesian_atlas import estimate_bayesian_atlas
from pydeformetrica.src.launch.estimate_deterministic_atlas import estimate_deterministic_atlas
from pydeformetrica.src.launch.estimate_geodesic_regression import estimate_geodesic_regression
from pydeformetrica.src.core.model_tools.deformations.exponential import Exponential
from pydeformetrica.src.core.model_tools.deformations.geodesic import Geodesic
from pydeformetrica.src.launch.estimate_longitudinal_atlas import estimate_longitudinal_atlas
from pydeformetrica.src.launch.estimate_longitudinal_registration import estimate_longitudinal_registration
from pydeformetrica.src.support.utilities.general_settings import Settings
from src.in_out.array_readers_and_writers import *
from pydeformetrica.src.support.kernels.kernel_functions import create_kernel
from pydeformetrica.src.core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from pydeformetrica.src.in_out.deformable_object_reader import DeformableObjectReader


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
            k = -1
            for model_xml_level2 in model_xml_level1:
                if model_xml_level2.tag.lower() == 'object':
                    k += 1
                    found_tag = False
                    for model_xml_level3 in model_xml_level2:
                        if model_xml_level3.tag.lower() == key.lower():
                            model_xml_level3.text = values[k]
                            found_tag = True
                    if not found_tag:
                        new_element_xml = et.SubElement(model_xml_level2, key)
                        new_element_xml.text = values[k]
    return model_xml_level0


def insert_model_xml_deformation_parameters_entry(model_xml_level0, key, value):
    for model_xml_level1 in model_xml_level0:
        if model_xml_level1.tag.lower() == 'deformation-parameters':
            found_tag = False
            for model_xml_level2 in model_xml_level1:
                if model_xml_level2.tag.lower() == key:
                    model_xml_level2.text = value
                    found_tag = True
            if not found_tag:
                new_element_xml = et.SubElement(model_xml_level1, key)
                new_element_xml.text = value
    return model_xml_level0


def estimate_geodesic_regression_for_subject(args):
    (i, general_settings, xml_parameters, regressions_output_path,
     global_full_dataset_filenames, global_full_visit_ages, global_full_subject_ids) = args

    Settings().initialize(general_settings)

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
    xml_parameters.state_file = None
    xml_parameters._further_initialization()

    # Adapt the global settings, for the custom output directory.
    Settings().output_dir = subject_regression_output_path
    Settings().state_file = os.path.join(Settings().output_dir, 'pydef_state.p')

    # Launch.
    model = estimate_geodesic_regression(xml_parameters)

    # Add the estimated momenta.
    return model.get_control_points(), model.get_momenta()


def reproject_momenta(source_control_points, source_momenta, target_control_points, kernel_width, kernel_type='torch'):
    kernel = create_kernel(kernel_type, kernel_width)
    source_control_points_torch = Variable(torch.from_numpy(source_control_points).type(Settings().tensor_scalar_type))
    source_momenta_torch = Variable(torch.from_numpy(source_momenta).type(Settings().tensor_scalar_type))
    target_control_points_torch = Variable(torch.from_numpy(target_control_points).type(Settings().tensor_scalar_type))
    target_momenta_torch = torch.potrs(
        kernel.convolve(source_control_points_torch, source_control_points_torch, source_momenta_torch),
        torch.potrf(kernel.get_kernel_matrix(target_control_points_torch)))
    # target_momenta_torch_bis = torch.mm(torch.inverse(kernel.get_kernel_matrix(target_control_points_torch)),
    #                                     kernel.convolve(source_control_points_torch, source_control_points_torch,
    #                                                     source_momenta_torch))
    return target_momenta_torch.data.cpu().numpy()


def parallel_transport(source_control_points, source_momenta, driving_momenta, kernel_width, kernel_type='torch'):
    source_control_points_torch = Variable(torch.from_numpy(source_control_points).type(Settings().tensor_scalar_type),
                                           requires_grad=(kernel_type == 'keops'))
    source_momenta_torch = Variable(torch.from_numpy(source_momenta).type(Settings().tensor_scalar_type))
    driving_momenta_torch = Variable(torch.from_numpy(driving_momenta).type(Settings().tensor_scalar_type))
    exponential = Exponential()
    exponential.set_kernel(create_kernel(kernel_type, kernel_width))
    exponential.number_of_time_points = 11
    exponential.set_use_rk2_for_shoot(True)  # Needed for parallel transport.
    exponential.set_initial_control_points(source_control_points_torch)
    exponential.set_initial_momenta(driving_momenta_torch)
    exponential.shoot()
    transported_control_points_torch = exponential.control_points_t[-1]
    transported_momenta_torch = exponential.parallel_transport(source_momenta_torch)[-1]
    return transported_control_points_torch.data.cpu().numpy(), transported_momenta_torch.data.cpu().numpy()


if __name__ == '__main__':

    print('')
    print('##############################')
    print('##### PyDeformetrica 1.0 #####')
    print('##############################')
    print('')

    """

    0]. Read command line, change directory, prepare preprocessing folder, read original xml parameters.
    """

    assert len(sys.argv) >= 4, 'Usage: ' + sys.argv[0] + " <model.xml> <data_set.xml> <optimization_parameters.xml> " \
                                                         "<optional --overwrite>"

    model_xml_path = sys.argv[1]
    dataset_xml_path = sys.argv[2]
    optimization_parameters_xml_path = sys.argv[3]

    preprocessings_folder = Settings().preprocessing_dir
    if not os.path.isdir(preprocessings_folder):
        os.mkdir(preprocessings_folder)

    # global_overwrite = True
    global_overwrite = False
    if len(sys.argv) > 4:
        if sys.argv[4] == '--overwrite':
            print('>> The script will overwrite the results from already performed initialization steps.')
            user_answer = input('>> Proceed with overwriting ? ([y]es / [n]o)')
            if str(user_answer).lower() in ['y', 'yes']:
                global_overwrite = True
            elif not str(user_answer).lower() in ['n', 'no']:
                print('>> Unexpected answer. Proceeding without overwriting.')
        else:
            msg = 'Unknown command-line option: "%s". Ignoring.' % sys.argv[4]
            warnings.warn(msg)

    # Read original longitudinal model xml parameters.
    xml_parameters = XmlParameters()
    xml_parameters._read_model_xml(model_xml_path)
    xml_parameters._read_dataset_xml(dataset_xml_path)
    xml_parameters._read_optimization_parameters_xml(optimization_parameters_xml_path)

    # Save some global parameters.
    global_full_dataset_filenames = xml_parameters.dataset_filenames
    global_full_visit_ages = xml_parameters.visit_ages
    global_full_subject_ids = xml_parameters.subject_ids

    global_objects_name, global_objects_name_extension \
        = create_template_metadata(xml_parameters.template_specifications)[1:3]

    global_user_specified_optimization_method = xml_parameters.optimization_method_type
    global_user_specified_number_of_threads = xml_parameters.number_of_threads

    global_dense_mode = xml_parameters.dense_mode
    global_deformation_kernel_type = xml_parameters.deformation_kernel_type
    global_deformation_kernel_width = xml_parameters.deformation_kernel_width

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

    """
    1]. Compute an atlas on the baseline data.
    ------------------------------------------
        The outputted template, control points and noise standard deviation will be used in the following
        geodesic regression and longitudinal registration, as well as an initialization for the longitudinal atlas.
        The template will always be used, i.e. the user-provided one is assumed to be a dummy, low-quality one.
        On the other hand, the estimated control points and noise standard deviation will only be used if the user did
        not provide those.
    """

    # atlas_type = 'Bayesian'
    atlas_type = 'Deterministic'

    atlas_output_path = os.path.join(preprocessings_folder, '1_atlas_on_baseline_data')
    if not global_overwrite and os.path.isdir(atlas_output_path):

        global_initial_template = DeformableMultiObject()
        global_initial_objects_template_path = []
        global_initial_objects_template_type = []
        reader = DeformableObjectReader()
        for object_id, object_specs in xml_parameters.template_specifications.items():
            extension = os.path.splitext(object_specs['filename'])[-1]
            filename = os.path.join('data', 'ForInitialization__Template_%s__FromAtlas%s' % (object_id, extension))
            object_type = object_specs['deformable_object_type'].lower()
            template_object = reader.create_object(filename, object_type)
            global_initial_template.object_list.append(template_object)
            global_initial_objects_template_path.append(filename)
            global_initial_objects_template_type.append(template_object.type.lower())
        global_initial_template.update()

        global_initial_template_data = global_initial_template.get_data()
        global_initial_control_points = read_2D_array(os.path.join(
            'data', 'ForInitialization__ControlPoints__FromAtlas.txt'))
        global_atlas_momenta = read_3D_array(os.path.join(
            atlas_output_path, 'DeterministicAtlas__EstimatedParameters__Momenta.txt'))

        model_xml_path = 'initialized_model.xml'

    else:
        print('[ estimate an atlas from baseline data ]')
        print('')

        # Initialization -----------------------------------------------------------------------------------------------
        # Clean folder.
        if os.path.isdir(atlas_output_path): shutil.rmtree(atlas_output_path)
        os.mkdir(atlas_output_path)

        # Adapt the xml parameters and update.
        xml_parameters.model_type = (atlas_type + 'Atlas').lower()
        # xml_parameters.optimization_method_type = 'ScipyLBFGS'.lower()
        xml_parameters.optimization_method_type = 'GradientAscent'.lower()
        xml_parameters.max_line_search_iterations = 20
        if True or xml_parameters.use_cuda:
            xml_parameters.number_of_threads = 1  # Problem to fix here. TODO.
            global_user_specified_number_of_threads = 1
        xml_parameters.print_every_n_iters = 1

        xml_parameters.initial_momenta = None

        xml_parameters.dataset_filenames = [[elt[0]] for elt in xml_parameters.dataset_filenames]
        xml_parameters.visit_ages = [[elt[0]] for elt in xml_parameters.visit_ages]

        xml_parameters._further_initialization()

        # Adapt the global settings, for the custom output directory.
        Settings().output_dir = atlas_output_path
        Settings().state_file = os.path.join(Settings().output_dir, 'pydef_state.p')

        # Launch and save the outputted noise standard deviation, for later use ----------------------------------------
        if atlas_type == 'Bayesian':
            model = estimate_bayesian_atlas(xml_parameters)
            global_objects_noise_std = [math.sqrt(elt) for elt in model.get_noise_variance()]

        elif atlas_type == 'Deterministic':
            model = estimate_deterministic_atlas(xml_parameters)
            global_objects_noise_std = [math.sqrt(elt) for elt in model.objects_noise_variance]

        else:
            raise RuntimeError('Unknown atlas type: "' + atlas_type + '"')

        # Export the results -------------------------------------------------------------------------------------------
        global_objects_name, global_objects_name_extension, original_objects_noise_variance = \
            create_template_metadata(xml_parameters.template_specifications)[1:4]

        global_initial_template = model.template
        global_initial_template_data = model.get_template_data()
        global_initial_control_points = model.get_control_points()
        global_atlas_momenta = model.get_momenta()

        global_initial_objects_template_path = []
        global_initial_objects_template_type = []
        for k, (object_name, object_name_extension, original_object_noise_variance) \
                in enumerate(zip(global_objects_name, global_objects_name_extension, original_objects_noise_variance)):

            # Save the template objects type.
            global_initial_objects_template_type.append(global_initial_template.object_list[k].type.lower())

            # Copy the estimated template to the data folder.
            estimated_template_path = os.path.join(
                atlas_output_path,
                atlas_type + 'Atlas__EstimatedParameters__Template_' + object_name + object_name_extension)
            global_initial_objects_template_path.append(os.path.join(
                'data', 'ForInitialization__Template_' + object_name + '__FromAtlas' + object_name_extension))
            shutil.copyfile(estimated_template_path, global_initial_objects_template_path[k])

            if global_initial_objects_template_type[k] == 'PolyLine'.lower():
                cmd = 'sed -i -- s/POLYGONS/LINES/g ' + global_initial_objects_template_path[k]
                os.system(cmd)  # Quite time-consuming.
                if os.path.isfile(global_initial_objects_template_path[k] + '--'):
                    os.remove(global_initial_objects_template_path[k] + '--')

            # Override the obtained noise standard deviation values, if it was already given by the user.
            if original_object_noise_variance > 0:
                global_objects_noise_std[k] = math.sqrt(original_object_noise_variance)

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
        doc = parseString((et.tostring(
            model_xml_level0).decode('utf-8').replace('\n', '').replace('\t', ''))).toprettyxml()
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

    # Check if the computations have been done already.
    regressions_output_path = os.path.join(preprocessings_folder, '2_individual_geodesic_regressions')
    if not global_overwrite and os.path.isdir(regressions_output_path):
        global_initial_momenta = read_3D_array(os.path.join('data', 'ForInitialization__Momenta__FromRegressions.txt'))

    # Check if an initial (longitudinal) momenta is available.
    elif global_initial_control_points_are_given and global_initial_momenta_are_given:
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
        if os.path.isdir(regressions_output_path): shutil.rmtree(regressions_output_path)
        os.mkdir(regressions_output_path)

        # Adapt the shared xml parameters.
        xml_parameters.model_type = 'Regression'.lower()
        # xml_parameters.optimization_method_type = 'ScipyLBFGS'.lower()
        xml_parameters.optimization_method_type = 'GradientAscent'.lower()
        xml_parameters.max_line_search_iterations = 10
        xml_parameters.initial_control_points = None
        xml_parameters.freeze_control_points = True
        xml_parameters.print_every_n_iters = 1

        # Launch -------------------------------------------------------------------------------------------------------
        Settings().number_of_threads = global_user_specified_number_of_threads
        # Multi-threaded version.
        if False and Settings().number_of_threads > 1 and not global_dense_mode:  # Non active for now.
            pool = Pool(processes=Settings().number_of_threads)
            args = [(i, Settings().serialize(), xml_parameters, regressions_output_path,
                     global_full_dataset_filenames, global_full_visit_ages, global_full_subject_ids)
                    for i in range(global_number_of_subjects)]
            global_initial_momenta = sum([elt[1] for elt in pool.map(estimate_geodesic_regression_for_subject, args)])
            pool.close()
            pool.join()

        # Single thread version.
        else:
            global_initial_momenta = np.zeros(global_initial_control_points.shape)
            for i in range(global_number_of_subjects):

                # Regression.
                regression_control_points, regression_momenta = estimate_geodesic_regression_for_subject((
                    i, Settings().serialize(), xml_parameters, regressions_output_path,
                    global_full_dataset_filenames, global_full_visit_ages, global_full_subject_ids))

                # Parallel transport of the estimated momenta.
                transported_regression_control_points, transported_regression_momenta = parallel_transport(
                    regression_control_points, regression_momenta, - global_atlas_momenta[i],
                    global_deformation_kernel_width, global_deformation_kernel_type)

                # Reprojection on the population control points.
                transported_and_reprojected_regression_momenta = reproject_momenta(
                    transported_regression_control_points, transported_regression_momenta,
                    global_initial_control_points, global_deformation_kernel_width, global_deformation_kernel_type)
                global_initial_momenta += transported_and_reprojected_regression_momenta

                # Saving this transported and reprojected momenta.
                path_to_subject_transported_and_reprojected_regression_momenta = os.path.join(
                    regressions_output_path, 'GeodesicRegression__subject_' + global_full_subject_ids[i],
                    'GeodesicRegression__EstimatedParameters__TransportedAndReprojectedMomenta.txt')
                np.savetxt(path_to_subject_transported_and_reprojected_regression_momenta,
                           transported_and_reprojected_regression_momenta)

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
    3]. Initializing heuristics for log-accelerations and onset ages.
    -----------------------------------------------------------------
        The individual accelerations are taken as the ratio of the regression momenta norm to the global one.
        The individual onset ages are computed as if all baseline ages were in correspondence.
    """

    print('')
    print('[ initializing heuristics for individual log-accelerations and onset ages ]')
    print('')

    kernel = create_kernel('torch', xml_parameters.deformation_kernel_width)

    global_initial_control_points_torch = torch.from_numpy(
        global_initial_control_points).type(Settings().tensor_scalar_type)

    global_initial_momenta_torch = torch.from_numpy(global_initial_momenta).type(Settings().tensor_scalar_type)
    global_initial_momenta_norm_squared = torch.dot(global_initial_momenta_torch.view(-1), kernel.convolve(
        global_initial_control_points_torch, global_initial_control_points_torch,
        global_initial_momenta_torch).view(-1))

    heuristic_initial_onset_ages = []
    heuristic_initial_log_accelerations = []
    for i in range(global_number_of_subjects):

        # Heuristic for the initial onset age.
        subject_mean_observation_age = np.mean(np.array(global_full_visit_ages[i]))
        heuristic_initial_onset_ages.append(subject_mean_observation_age)

        # Heuristic for the initial log-acceleration.
        path_to_subject_transported_and_reprojected_regression_momenta = os.path.join(
            regressions_output_path, 'GeodesicRegression__subject_' + global_full_subject_ids[i],
            'GeodesicRegression__EstimatedParameters__TransportedAndReprojectedMomenta.txt')
        subject_regression_momenta = read_3D_array(path_to_subject_transported_and_reprojected_regression_momenta)
        subject_regression_momenta_torch = torch.from_numpy(
            subject_regression_momenta).type(Settings().tensor_scalar_type)

        subject_regression_momenta_scalar_product_with_population_momenta = torch.dot(
            global_initial_momenta_torch.view(-1), kernel.convolve(
                global_initial_control_points_torch, global_initial_control_points_torch,
                subject_regression_momenta_torch).view(-1)).cpu().numpy()

        if subject_regression_momenta_scalar_product_with_population_momenta <= 0.0:
            msg = 'Subject %s seems to evolve against the population: scalar_product = %.3E.' % \
                  (global_full_subject_ids[i],
                   Decimal(float(subject_regression_momenta_scalar_product_with_population_momenta)))
            warnings.warn(msg)
            print('>> ' + msg)
            heuristic_initial_log_accelerations.append(1.0)  # Neutral initialization.
        else:
            heuristic_initial_log_accelerations.append(
                math.log(math.sqrt(subject_regression_momenta_scalar_product_with_population_momenta
                         / global_initial_momenta_norm_squared)))

    heuristic_initial_onset_ages = np.array(heuristic_initial_onset_ages)
    heuristic_initial_log_accelerations = np.array(heuristic_initial_log_accelerations)

    # Rescaling the initial momenta according to the mean of the acceleration factors.
    mean_log_acceleration = np.mean(heuristic_initial_log_accelerations)
    heuristic_initial_log_accelerations -= mean_log_acceleration
    global_initial_momenta *= math.exp(mean_log_acceleration)

    print('>> Estimated random effect statistics:')
    print('\t\t onset_ages        =\t%.3f\t[ mean ]\t+/-\t%.4f\t[std]' %
          (np.mean(heuristic_initial_onset_ages), np.std(heuristic_initial_onset_ages)))
    print('\t\t log_accelerations =\t%.4f\t[ mean ]\t+/-\t%.4f\t[std]' %
          (np.mean(heuristic_initial_log_accelerations), np.std(heuristic_initial_log_accelerations)))

    # Export the results -----------------------------------------------------------------------------------------------
    # Initial momenta.
    global_initial_momenta_path = os.path.join('data', 'ForInitialization__Momenta__RescaledWithHeuristics.txt')
    np.savetxt(global_initial_momenta_path, global_initial_momenta)

    # Onset ages.
    heuristic_initial_onset_ages_path = os.path.join(
        'data', 'ForInitialization__OnsetAges__FromHeuristic.txt')
    np.savetxt(heuristic_initial_onset_ages_path, heuristic_initial_onset_ages)

    # Log-accelerations.
    heuristic_initial_log_accelerations_path = os.path.join(
        'data', 'ForInitialization__LogAccelerations__FromHeuristic.txt')
    np.savetxt(heuristic_initial_log_accelerations_path, heuristic_initial_log_accelerations)

    # Modify the original model.xml file accordingly.
    model_xml_level0 = et.parse(model_xml_path).getroot()
    model_xml_level0 = insert_model_xml_level1_entry(
        model_xml_level0, 'initial-momenta', global_initial_momenta_path)
    model_xml_level0 = insert_model_xml_level1_entry(
        model_xml_level0, 'initial-onset-ages', heuristic_initial_onset_ages_path)
    model_xml_level0 = insert_model_xml_level1_entry(
        model_xml_level0, 'initial-log-accelerations', heuristic_initial_log_accelerations_path)
    model_xml_path = 'initialized_model.xml'
    doc = parseString((et.tostring(
        model_xml_level0).decode('utf-8').replace('\n', '').replace('\t', ''))).toprettyxml()
    np.savetxt(model_xml_path, [doc], fmt='%s')

    """
    4]. Shoot from the average baseline age to the global average.
    --------------------------------------------------------------
        New values are obtained for the template, control points, and (longitudinal) momenta.
        Skipped if initial control points and momenta were given.
    """

    if not global_initial_control_points_are_given and not global_initial_momenta_are_given:

        print('')
        print('[ shoot from the average baseline age to the global average ]')

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
        geodesic.set_use_rk2_for_shoot(xml_parameters.use_rk2_for_shoot)
        geodesic.set_use_rk2_for_flow(xml_parameters.use_rk2_for_flow)
        geodesic.set_t0(global_tmin)
        geodesic.set_tmin(global_tmin)
        geodesic.set_tmax(global_t0)

        # Set the template, control points and momenta and update.
        geodesic.set_template_points_t0(
            {key: Variable(torch.from_numpy(value).type(Settings().tensor_scalar_type), requires_grad=False)
             for key, value in global_initial_template.get_points().items()})
        if Settings().dense_mode:
            geodesic.set_control_points_t0(geodesic.get_template_data_t0())
        else:
            geodesic.set_control_points_t0(Variable(torch.from_numpy(
                global_initial_control_points).type(Settings().tensor_scalar_type),
                                                    requires_grad=(geodesic.get_kernel_type() == 'keops')))
        geodesic.set_momenta_t0(Variable(torch.from_numpy(
            global_initial_momenta).type(Settings().tensor_scalar_type), requires_grad=False))
        geodesic.update()

        # Adapt the global settings, for the custom output directory.
        Settings().output_dir = shooting_output_path

        # Write.
        geodesic.write('Shooting', global_objects_name, global_objects_name_extension, global_initial_template,
                       {key: Variable(torch.from_numpy(value).type(Settings().tensor_scalar_type), requires_grad=False)
                        for key, value in global_initial_template_data.items()},
                       write_adjoint_parameters=True)

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
                'data', 'ForInitialization__Template_' + object_name + '__FromAtlasAndShooting' + object_name_extension)
            shutil.copyfile(shooted_template_path, global_initial_objects_template_path[k])

            if global_initial_objects_template_type[k] == 'PolyLine'.lower():
                cmd = 'sed -i -- s/POLYGONS/LINES/g ' + global_initial_objects_template_path[k]
                os.system(cmd)  # Quite time-consuming.
                if os.path.isfile(global_initial_objects_template_path[k] + '--'):
                    os.remove(global_initial_objects_template_path[k] + '--')

        # Control points.
        shooted_control_points_path = os.path.join(
            shooting_output_path, 'Shooting__GeodesicFlow__ControlPoints__tp_' + str(number_of_timepoints)
                                  + ('__age_%.2f' % global_t0) + '.txt')
        global_initial_control_points_path = os.path.join('data',
                                                          'ForInitialization__ControlPoints__FromAtlasAndShooting.txt')
        shutil.copyfile(shooted_control_points_path, global_initial_control_points_path)

        # Momenta.
        shooted_momenta_path = os.path.join(
            shooting_output_path, 'Shooting__GeodesicFlow__Momenta__tp_' + str(number_of_timepoints)
                                  + ('__age_%.2f' % global_t0) + '.txt')
        global_initial_momenta_path = os.path.join('data', 'ForInitialization__Momenta__FromRegressionsAndShooting.txt')
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
    5]. Tangent-space ICA on the individual momenta outputted by the atlas estimation.
    ----------------------------------------------------------------------------------
        Those momenta are first projected on the space orthogonal to the initial (longitudinal) momenta.
        Skipped if initial control points and modulation matrix were specified.
    """

    # Check if an initial (longitudinal) momenta is available.
    if not (global_initial_control_points_are_given and global_initial_modulation_matrix_is_given):

        print('')
        print('[ tangent-space ICA on the projected individual momenta ]')
        print('')

        # Warning.
        if not global_initial_control_points_are_given and global_initial_modulation_matrix_is_given:
            msg = 'Initial modulation matrix is given but not the corresponding initial control points. ' \
                  'This given initial modulation matrix will be ignored, and overridden by a ICA-based heuristic.'
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
        kernel = create_kernel('torch', xml_parameters.deformation_kernel_width)

        Km = np.dot(K, global_initial_momenta.ravel())
        mKm = np.dot(global_initial_momenta.ravel().transpose(), Km)

        w = []
        for i in range(momenta.shape[0]):
            w.append(momenta[i].ravel() - np.dot(momenta[i].ravel(), Km) / mKm * global_initial_momenta.ravel())
        w = np.array(w)

        # Dimensionality reduction.
        if xml_parameters.number_of_sources is not None:
            number_of_sources = xml_parameters.number_of_sources
        elif xml_parameters.initial_modulation_matrix is not None:
            number_of_sources = read_2D_array(xml_parameters.initial_modulation_matrix).shape[1]
        else:
            number_of_sources = 4
            print('>> No initial modulation matrix given, neither a number of sources. '
                  'The latter will be ARBITRARILY defaulted to 4.')

        ica = FastICA(n_components=number_of_sources, max_iter=50000)
        global_initial_sources = ica.fit_transform(w)
        global_initial_modulation_matrix = ica.mixing_

        # Rescale.
        for s in range(number_of_sources):
            std = np.std(global_initial_sources[:, s])
            global_initial_sources[:, s] /= std
            global_initial_modulation_matrix[:, s] *= std

        # Print.
        residuals = []
        for i in range(global_number_of_subjects):
            residuals.append(w[i] - np.dot(global_initial_modulation_matrix, global_initial_sources[i]))
        mean_relative_residual = np.mean(np.absolute(np.array(residuals))) / np.mean(np.absolute(w))
        print('>> Mean relative residual: %.3f %%.' % (100 * mean_relative_residual))

        # Save.
        global_initial_modulation_matrix_path = \
            os.path.join('data', 'ForInitialization__ModulationMatrix__FromICA.txt')
        np.savetxt(global_initial_modulation_matrix_path, global_initial_modulation_matrix)

        global_initial_sources_path = os.path.join('data', 'ForInitialization__Sources__FromICA.txt')
        np.savetxt(global_initial_sources_path, global_initial_sources)

        # Modify the original model.xml file accordingly.
        model_xml_level0 = et.parse(model_xml_path).getroot()
        model_xml_level0 = insert_model_xml_level1_entry(
            model_xml_level0, 'initial-modulation-matrix', global_initial_modulation_matrix_path)
        model_xml_level0 = insert_model_xml_level1_entry(
            model_xml_level0, 'initial-sources', global_initial_sources_path)
        model_xml_path = 'initialized_model.xml'
        doc = parseString(
            (et.tostring(model_xml_level0).decode('utf-8').replace('\n', '').replace('\t', ''))).toprettyxml()
        np.savetxt(model_xml_path, [doc], fmt='%s')

        print('>> Estimated random effect statistics:')
        print('\t\t sources =\t%.3f\t[ mean ]\t+/-\t%.4f\t[std]' %
              (np.mean(global_initial_sources), np.std(global_initial_sources)))

    """
    6]. Longitudinal registration of all target subjects.
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
    if os.path.isdir(registration_output_path):
        if global_overwrite:
            shutil.rmtree(registration_output_path)
        elif not os.path.isdir(os.path.join(registration_output_path, 'tmp')):
            registrations = os.listdir(registration_output_path)
            if len(registrations) > 0:
                shutil.rmtree(os.path.join(registration_output_path, os.listdir(registration_output_path)[-1]))
    if not os.path.isdir(registration_output_path): os.mkdir(registration_output_path)

    # Read the current longitudinal model xml parameters.
    xml_parameters = XmlParameters()
    xml_parameters._read_model_xml(model_xml_path)
    xml_parameters._read_dataset_xml(dataset_xml_path)
    xml_parameters._read_optimization_parameters_xml(optimization_parameters_xml_path)

    # Adapt the xml parameters and update.
    xml_parameters.model_type = 'LongitudinalRegistration'.lower()
    # xml_parameters.optimization_method_type = 'ScipyPowell'.lower()
    xml_parameters.optimization_method_type = 'ScipyLBFGS'.lower()
    xml_parameters.convergence_tolerance = 1e-3
    xml_parameters.print_every_n_iters = 1
    xml_parameters._further_initialization()

    # Adapt the global settings, for the custom output directory.
    Settings().output_dir = registration_output_path

    # Launch.
    estimate_longitudinal_registration(xml_parameters, overwrite=global_overwrite)

    # Load results.
    estimated_onset_ages_path = os.path.join(
        registration_output_path, 'LongitudinalRegistration__EstimatedParameters__OnsetAges.txt')
    estimated_log_accelerations_path = os.path.join(
        registration_output_path, 'LongitudinalRegistration__EstimatedParameters__LogAccelerations.txt')
    estimated_sources_path = os.path.join(
        registration_output_path, 'LongitudinalRegistration__EstimatedParameters__Sources.txt')

    global_onset_ages = read_2D_array(estimated_onset_ages_path)
    global_log_accelerations = read_2D_array(estimated_log_accelerations_path)
    global_sources = read_2D_array(estimated_sources_path)

    # Rescaling the initial momenta according to the mean of the acceleration factors.
    mean_log_acceleration = np.mean(global_log_accelerations)
    global_log_accelerations -= mean_log_acceleration
    global_initial_momenta *= math.exp(mean_log_acceleration)

    print('')
    print('>> Estimated random effect statistics:')
    print('\t\t onset_ages        =\t%.3f\t[ mean ]\t+/-\t%.4f\t[std]' %
          (np.mean(heuristic_initial_onset_ages), np.std(heuristic_initial_onset_ages)))
    print('\t\t log_accelerations =\t%.4f\t[ mean ]\t+/-\t%.4f\t[std]' %
          (np.mean(heuristic_initial_log_accelerations), np.std(heuristic_initial_log_accelerations)))
    print('\t\t sources           =\t%.4f\t[ mean ]\t+/-\t%.4f\t[std]' %
          (np.mean(global_sources), np.std(global_sources)))

    # Copy the output individual effects into the data folder.
    # Initial momenta.
    global_initial_momenta_path = os.path.join(
        'data', 'ForInitialization__Momenta__RescaledWithLongitudinalRegistration.txt')
    np.savetxt(global_initial_momenta_path, global_initial_momenta)

    # Onset ages.
    global_initial_onset_ages_path = os.path.join(
        'data', 'ForInitialization__OnsetAges__FromLongitudinalRegistration.txt')
    shutil.copyfile(estimated_onset_ages_path, global_initial_onset_ages_path)

    # Log-accelerations.
    global_initial_log_accelerations_path = os.path.join(
        'data', 'ForInitialization__LogAccelerations__FromLongitudinalRegistration.txt')
    np.savetxt(global_initial_log_accelerations_path, global_log_accelerations)

    # Sources.
    global_initial_sources_path = os.path.join(
        'data', 'ForInitialization__Sources__FromLongitudinalRegistration.txt')
    shutil.copyfile(estimated_sources_path, global_initial_sources_path)

    # Modify the original model.xml file accordingly.
    model_xml_level0 = et.parse(model_xml_path).getroot()
    model_xml_level0 = insert_model_xml_level1_entry(
        model_xml_level0, 'initial-momenta', global_initial_momenta_path)
    model_xml_level0 = insert_model_xml_level1_entry(
        model_xml_level0, 'initial-onset-ages', global_initial_onset_ages_path)
    model_xml_level0 = insert_model_xml_level1_entry(
        model_xml_level0, 'initial-log-accelerations', global_initial_log_accelerations_path)
    model_xml_level0 = insert_model_xml_level1_entry(
        model_xml_level0, 'initial-sources', global_initial_sources_path)
    model_xml_path = 'initialized_model.xml'
    doc = parseString((et.tostring(
        model_xml_level0).decode('utf-8').replace('\n', '').replace('\t', ''))).toprettyxml()
    np.savetxt(model_xml_path, [doc], fmt='%s')

    """
    7]. Gradient-based optimization on population parameters.
    ---------------------------------------------------------
        Ignored if the user-specified optimization method is not the MCMC-SAEM.
    """

    if global_user_specified_optimization_method.lower() == 'McmcSaem'.lower():

        print('')
        print('[ longitudinal atlas estimation with the GradientAscent optimizer ]')
        print('')

        # Prepare and launch the longitudinal atlas estimation ---------------------------------------------------------
        # Clean folder.
        longitudinal_atlas_output_path = os.path.join(
            preprocessings_folder, '5_longitudinal_atlas_with_gradient_ascent')
        if os.path.isdir(longitudinal_atlas_output_path): shutil.rmtree(longitudinal_atlas_output_path)
        os.mkdir(longitudinal_atlas_output_path)

        # Read the current longitudinal model xml parameters, adapt them and update.
        xml_parameters = XmlParameters()
        xml_parameters._read_model_xml(model_xml_path)
        xml_parameters._read_dataset_xml(dataset_xml_path)
        xml_parameters._read_optimization_parameters_xml(optimization_parameters_xml_path)
        xml_parameters.optimization_method_type = 'GradientAscent'.lower()
        xml_parameters.optimized_log_likelihood = 'class2'.lower()
        xml_parameters.max_line_search_iterations = 20
        xml_parameters.print_every_n_iters = 1
        xml_parameters._further_initialization()

        # Adapt the global settings, for the custom output directory.
        Settings().output_dir = longitudinal_atlas_output_path
        Settings().state_file = os.path.join(longitudinal_atlas_output_path, 'pydef_state.p')

        # Launch.
        model = estimate_longitudinal_atlas(xml_parameters)

        # Export the results -------------------------------------------------------------------------------------------
        model_xml_level0 = et.parse(model_xml_path).getroot()

        # Template.
        for k, (object_name, object_name_extension) in enumerate(zip(global_objects_name,
                                                                     global_objects_name_extension)):
            estimated_template_path = os.path.join(
                longitudinal_atlas_output_path,
                'LongitudinalAtlas__EstimatedParameters__Template_%s__tp_%d__age_%.2f%s' %
                (object_name,
                 model.spatiotemporal_reference_frame.geodesic.backward_exponential.number_of_time_points - 1,
                 model.get_reference_time(), object_name_extension))
            global_initial_objects_template_path[k] = os.path.join(
                'data',
                'ForInitialization__Template_%s__FromLongitudinalAtlas%s' % (object_name, object_name_extension))
            shutil.copyfile(estimated_template_path, global_initial_objects_template_path[k])

            if global_initial_objects_template_type[k] == 'PolyLine'.lower():
                cmd = 'sed -i -- s/POLYGONS/LINES/g ' + global_initial_objects_template_path[k]
                os.system(cmd)  # Quite time-consuming.
                if os.path.isfile(global_initial_objects_template_path[k] + '--'):
                    os.remove(global_initial_objects_template_path[k] + '--')

        model_xml_level0 = insert_model_xml_template_spec_entry(
            model_xml_level0, 'filename', global_initial_objects_template_path)

        # Control points.
        estimated_control_points_path = os.path.join(
            longitudinal_atlas_output_path, 'LongitudinalAtlas__EstimatedParameters__ControlPoints.txt')
        global_initial_control_points_path = os.path.join(
            'data', 'ForInitialization__ControlPoints__FromLongitudinalAtlas.txt')
        shutil.copyfile(estimated_control_points_path, global_initial_control_points_path)
        model_xml_level0 = insert_model_xml_level1_entry(
            model_xml_level0, 'initial-control-points', global_initial_control_points_path)

        # Momenta.
        estimated_momenta_path = os.path.join(
            longitudinal_atlas_output_path, 'LongitudinalAtlas__EstimatedParameters__Momenta.txt')
        global_initial_momenta_path = os.path.join(
            'data', 'ForInitialization__Momenta__FromLongitudinalAtlas.txt')
        shutil.copyfile(estimated_momenta_path, global_initial_momenta_path)
        model_xml_level0 = insert_model_xml_level1_entry(
            model_xml_level0, 'initial-momenta', global_initial_momenta_path)

        # Modulation matrix.
        estimated_modulation_matrix_path = os.path.join(
            longitudinal_atlas_output_path, 'LongitudinalAtlas__EstimatedParameters__ModulationMatrix.txt')
        global_initial_modulation_matrix_path = os.path.join(
            'data', 'ForInitialization__ModulationMatrix__FromLongitudinalAtlas.txt')
        shutil.copyfile(estimated_modulation_matrix_path, global_initial_modulation_matrix_path)
        model_xml_level0 = insert_model_xml_level1_entry(
            model_xml_level0, 'initial-modulation-matrix', global_initial_modulation_matrix_path)

        # Reference time.
        estimated_reference_time_path = os.path.join(
            longitudinal_atlas_output_path, 'LongitudinalAtlas__EstimatedParameters__ReferenceTime.txt')
        global_initial_reference_time = np.loadtxt(estimated_reference_time_path)
        model_xml_level0 = insert_model_xml_deformation_parameters_entry(
            model_xml_level0, 't0', '%.4f' % global_initial_reference_time)

        # Time-shift variance.
        estimated_time_shift_std_path = os.path.join(
            longitudinal_atlas_output_path, 'LongitudinalAtlas__EstimatedParameters__TimeShiftStd.txt')
        global_initial_time_shift_std = np.loadtxt(estimated_time_shift_std_path)
        model_xml_level0 = insert_model_xml_level1_entry(
            model_xml_level0, 'initial-time-shift-std', '%.4f' % global_initial_time_shift_std)

        # Log-acceleration variance.
        estimated_log_acceleration_std_path = os.path.join(
            longitudinal_atlas_output_path, 'LongitudinalAtlas__EstimatedParameters__LogAccelerationStd.txt')
        global_initial_log_acceleration_std = np.loadtxt(estimated_log_acceleration_std_path)
        model_xml_level0 = insert_model_xml_level1_entry(
            model_xml_level0, 'initial-log-acceleration-std', '%.4f' % global_initial_log_acceleration_std)

        # Noise variance.
        global_initial_noise_variance = model.get_noise_variance()
        global_initial_noise_std_string = ['{:.4f}'.format(math.sqrt(elt)) for elt in global_initial_noise_variance]
        model_xml_level0 = insert_model_xml_template_spec_entry(
            model_xml_level0, 'noise-std', global_initial_noise_std_string)

        # Onset ages.
        estimated_onset_ages_path = os.path.join(longitudinal_atlas_output_path,
                                                 'LongitudinalAtlas__EstimatedParameters__OnsetAges.txt')
        global_initial_onset_ages_path = os.path.join('data', 'ForInitialization__OnsetAges__FromLongitudinalAtlas.txt')
        shutil.copyfile(estimated_onset_ages_path, global_initial_onset_ages_path)
        model_xml_level0 = insert_model_xml_level1_entry(
            model_xml_level0, 'initial-onset-ages', global_initial_onset_ages_path)

        # Log-accelerations.
        estimated_log_accelerations_path = os.path.join(
            longitudinal_atlas_output_path, 'LongitudinalAtlas__EstimatedParameters__LogAccelerations.txt')
        global_initial_log_accelerations_path = os.path.join(
            'data', 'ForInitialization__LogAccelerations__FromLongitudinalAtlas.txt')
        shutil.copyfile(estimated_log_accelerations_path, global_initial_log_accelerations_path)
        model_xml_level0 = insert_model_xml_level1_entry(
            model_xml_level0, 'initial-log-accelerations', global_initial_log_accelerations_path)

        # Sources.
        estimated_sources_path = os.path.join(longitudinal_atlas_output_path,
                                              'LongitudinalAtlas__EstimatedParameters__Sources.txt')
        global_initial_sources_path = os.path.join('data', 'ForInitialization__Sources__FromLongitudinalAtlas.txt')
        shutil.copyfile(estimated_sources_path, global_initial_sources_path)
        model_xml_level0 = insert_model_xml_level1_entry(
            model_xml_level0, 'initial-sources', global_initial_sources_path)

        # Finalization.
        model_xml_path = 'initialized_model.xml'
        doc = parseString(
            (et.tostring(model_xml_level0).decode('utf-8').replace('\n', '').replace('\t', ''))).toprettyxml()
        np.savetxt(model_xml_path, [doc], fmt='%s')
