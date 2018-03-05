import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

import shutil
import xml.etree.ElementTree as et

from pydeformetrica.src.in_out.xml_parameters import XmlParameters
from pydeformetrica.src.launch.estimate_geodesic_regression import estimate_geodesic_regression
from pydeformetrica.src.support.utilities.general_settings import Settings
from src.in_out.array_readers_and_writers import *
import xml.etree.ElementTree as et
from xml.dom.minidom import parseString
from pydeformetrica.src.launch.estimate_longitudinal_metric_model import estimate_longitudinal_metric_model


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
    # Settings().state_file = None

    # Launch.
    estimate_geodesic_regression(xml_parameters)

    # Add the estimated momenta.
    return read_3D_array(os.path.join(
        subject_regression_output_path, 'GeodesicRegression__EstimatedParameters__Momenta.txt'))


if __name__ == '__main__':

    print('')
    print('##############################')
    print('##### PyDeformetrica 1.0 #####')
    print('##############################')

    print('')

    assert len(sys.argv) == 4, 'Usage: ' + sys.argv[0] + " <model.xml> <data_set.xml> <optimization_parameters.xml> "

    model_xml_path = sys.argv[1]
    dataset_xml_path = sys.argv[2]
    optimization_parameters_xml_path = sys.argv[3]

    preprocessings_folder = Settings().preprocessing_dir
    if not os.path.isdir(preprocessings_folder):
        pass

    mode_descent_output_path = os.path.join(preprocessings_folder, '1_gradient_descent_on_the_mode')
    # To perform this gradient descent, we use the iniialization heuristic, starting from
    # a flat metric and linear regressions one each subject

    # Read original longitudinal model xml parameters.
    xml_parameters = XmlParameters()
    xml_parameters._read_model_xml(model_xml_path)
    xml_parameters._read_dataset_xml(dataset_xml_path)
    xml_parameters._read_optimization_parameters_xml(optimization_parameters_xml_path)

    xml_parameters.optimization_method_type = 'GradientAscent'.lower()
    xml_parameters.scale_initial_step_size = True
    xml_parameters.initialization_heuristic = True
    xml_parameters.max_iterations = 50

    xml_parameters.output_dir = mode_descent_output_path
    Settings().set_output_dir(mode_descent_output_path)

    print(" >>> Performing gradient descent on the mode.")

    estimate_longitudinal_metric_model(xml_parameters)

    # Now that this is done, we create the right xml parameters file for the actual computation.
    # We already have the dataset_xml file: it's ok.
    # We already have the optimization_parameters file.
    # We must create a model.xml file.

    model_xml = et.Element('data-set')
    model_xml.set('deformetrica-min-version', "3.0.0")

    model_type = et.SubElement(model_xml, 'model-type')
    model_type.text = "LongitudinalMetricLearning"

    estimated_alphas = np.loadtxt(os.path.join(mode_descent_output_path, 'LongitudinalMetricModel_alphas.txt'))
    estimated_onset_ages = np.loadtxt(os.path.join(mode_descent_output_path, 'LongitudinalMetricModel_onset_ages.txt'))

    initial_time_shift_std = et.SubElement(model_xml, 'initial-time-shift-std')
    initial_time_shift_std.text = str(np.std(estimated_onset_ages))

    initial_log_acceleration_std = et.SubElement(model_xml, 'initial-log-acceleration-std')
    initial_log_acceleration_std.text = str(np.std(np.log(estimated_alphas)))

    deformation_parameters = et.SubElement(model_xml, 'deformation-parameters')

    exponential_type = et.SubElement(deformation_parameters, 'exponential-type')
    exponential_type.text = xml_parameters.exponential_type

    concentration_of_timepoints = et.SubElement(deformation_parameters,
                                                'concentration-of-timepoints')
    concentration_of_timepoints.text = str(xml_parameters.concentration_of_time_points)

    estimated_fixed_effects = np.load(os.path.join(mode_descent_output_path,
                                                   'LongitudinalMetricModel_all_fixed_effects.npy'))[
        ()]


    metric_parameters_file = et.SubElement(deformation_parameters,
                                                'metric-parameters-file')
    metric_parameters_file.text = os.path.join(mode_descent_output_path, 'LongitudinalMetricModel_metric_parameters.txt')

    t0 = et.SubElement(deformation_parameters, 't0')
    t0.text = str(estimated_fixed_effects['reference_time'])

    v0 = et.SubElement(deformation_parameters, 'v0')
    v0.text = str(estimated_fixed_effects['v0'][0])

    p0 = et.SubElement(deformation_parameters, 'p0')
    p0.text = str(estimated_fixed_effects['p0'][0])

    initial_onset_ages = et.SubElement(model_xml, 'initial-onset-ages')
    initial_onset_ages.text = os.path.join(mode_descent_output_path,
                                           "LongitudinalMetricModel_onset_ages.txt")

    initial_log_accelerations = et.SubElement(model_xml, 'initial-log-accelerations')
    initial_log_accelerations.text = os.path.join(mode_descent_output_path,
                                                  "LongitudinalMetricModel_log_accelerations.txt")


    model_xml_path = 'model_after_initialization.xml'
    doc = parseString((et.tostring(model_xml).decode('utf-8').replace('\n', '').replace('\t', ''))).toprettyxml()
    np.savetxt(model_xml_path, [doc], fmt='%s')

    # Or maybe run the estimation right after !

    # We also need to modify the proposition std in the optimization parameters ?