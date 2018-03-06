import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

import shutil
import xml.etree.ElementTree as et

from pydeformetrica.src.in_out.xml_parameters import XmlParameters
from pydeformetrica.src.support.utilities.general_settings import Settings
from src.in_out.array_readers_and_writers import *
import xml.etree.ElementTree as et
from xml.dom.minidom import parseString
from pydeformetrica.src.launch.estimate_longitudinal_metric_model import estimate_longitudinal_metric_model
from sklearn import datasets, linear_model
from pydeformetrica.src.in_out.dataset_functions import read_and_create_scalar_dataset

def _smart_initialization_individual_effects(dataset):
    """
    least_square regression for each subject, so that yi = ai * t + bi
    output is the list of ais and bis
    this proceeds as if the initialization for the geodesic is a straight line
    """
    print("Performing initial least square regressions on the subjects, for initialization purposes.")

    number_of_subjects = dataset.number_of_subjects

    ais = []
    bis = []

    for i in range(number_of_subjects):

        # Special case of a single observation for the subject
        if len(dataset.times[i]) <= 1:
            ais.append(1.)
            bis.append(0.)

        least_squares = linear_model.LinearRegression()
        least_squares.fit(dataset.times[i].reshape(-1, 1), dataset.deformable_objects[i].data.numpy().reshape(-1, 1))

        ais.append(max(0.001, least_squares.coef_[0][0]))
        bis.append(least_squares.intercept_[0])

        #if the slope is negative, we change it to 0.03, arbitrarily...

    # Ideally replace this by longitudinal registrations on the initial metric ! (much more expensive though)

    return ais, bis

def _smart_initialization(dataset):
    ais, bis = _smart_initialization_individual_effects(dataset)
    reference_time = np.mean([np.mean(times_i) for times_i in dataset.times])
    average_a = np.mean(ais)
    average_b = np.mean(bis)
    alphas = []
    onset_ages = []
    for i in range(len(ais)):
        alphas.append(max(0.2, min(ais[i] / average_a, 2.5)))  # Arbitrary bounds for a sane initialization
        onset_ages.append(max(reference_time - 15,
                              min(reference_time + 15, (reference_time * average_a + average_b - bis[i]) / ais[i])))
    # p0 = average_a * reference_time + average_b

    p0 = 0
    for i in range(dataset.number_of_subjects):
        p0 += np.mean(dataset.deformable_objects[i].data.numpy())
    p0 /= dataset.number_of_subjects

    return reference_time, average_a, p0, np.array(onset_ages), np.array(alphas)


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

    # Read original longitudinal model xml parameters.
    xml_parameters = XmlParameters()
    xml_parameters._read_model_xml(model_xml_path)
    xml_parameters._read_dataset_xml(dataset_xml_path)
    xml_parameters._read_optimization_parameters_xml(optimization_parameters_xml_path)

    """
    1) Simple heuristic for initializing everything but the sources and the modulation matrix.
    """

    smart_initialization_output_path = os.path.join(preprocessings_folder, '1_smart_initialization')
    Settings().output_dir = smart_initialization_output_path

    # We call the smart initialization. We need to instantiate the dataset first.

    dataset = read_and_create_scalar_dataset(xml_parameters)

    reference_time, average_a, p0, onset_ages, alphas = _smart_initialization(dataset)

    # We save the onset ages and alphas.
    # We then set the right path in the xml_parameters, for the proper initialization.
    write_2D_array(np.log(alphas), "SmartInitialization_log_accelerations.txt")
    write_2D_array(onset_ages, "SmartInitialization_onset_ages.txt")

    xml_parameters.initial_onset_alphas = os.path.join(smart_initialization_output_path, "SmartInitialization_onset_ages.txt")
    xml_parameters.initial_log_accelerations = os.path.join(smart_initialization_output_path, "SmartInitialization_log_accelerations.txt")
    xml_parameters.t0 = reference_time
    xml_parameters.v0 = average_a
    xml_parameters.p0 = p0

    """
    2) Gradient descent on the mode
    """

    mode_descent_output_path = os.path.join(preprocessings_folder, '1_gradient_descent_on_the_mode')
    # To perform this gradient descent, we use the iniialization heuristic, starting from
    # a flat metric and linear regressions one each subject

    xml_parameters.optimization_method_type = 'GradientAscent'.lower()
    xml_parameters.scale_initial_step_size = True
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

    if xml_parameters.exponential_type in ['parametric']: # otherwise it's not saved !
        metric_parameters_file = et.SubElement(deformation_parameters,
                                                    'metric-parameters-file')
        metric_parameters_file.text = os.path.join(mode_descent_output_path, 'LongitudinalMetricModel_metric_parameters.txt')

    if xml_parameters.number_of_sources > 0:
        initial_sources_file = et.SubElement(model_xml, 'initial-sources')
        initial_sources_file.text = os.path.join(mode_descent_output_path, 'LongitudinalMetricModel_sources.txt')

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
