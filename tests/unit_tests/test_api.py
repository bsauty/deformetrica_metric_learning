import logging
import os
import platform
import time
import unittest

import torch

from api.deformetrica import Deformetrica
from support.utilities import adni_extract_from_file_name
from unit_tests import example_data_dir, sandbox_data_dir, functional_tests_data_dir

logging.basicConfig(level=logging.DEBUG)


class API(unittest.TestCase):
    def setUp(self):
        self.deformetrica = Deformetrica(output_dir=os.path.join(os.path.dirname(__file__), 'output'))
        self.has_estimator_callback_been_called = False
        self.current_iteration = 0

    def __estimator_callback(self, status_dict):
        self.assertTrue('current_iteration' in status_dict)
        self.assertTrue('current_log_likelihood' in status_dict)
        self.assertTrue('current_attachment' in status_dict)
        self.assertTrue('current_regularity' in status_dict)
        self.assertTrue('gradient' in status_dict)
        self.current_iteration = status_dict['current_iteration']
        self.has_estimator_callback_been_called = True
        return True

    def __estimator_callback_stop(self, status_dict):
        self.__estimator_callback(status_dict)
        return False

    def test_api_version(self):
        from src import __version__
        print(__version__)
        self.assertIsNotNone(__version__)
        self.assertTrue(isinstance(__version__, str))

    def test_estimator_loop_stop(self):
        dataset_specifications = {
            'dataset_filenames': [
                [{'skull': example_data_dir + '/atlas/landmark/2d/skulls/data/skull_australopithecus.vtk'}],
                [{'skull': example_data_dir + '/atlas/landmark/2d/skulls/data/skull_erectus.vtk'}],
                [{'skull': example_data_dir + '/atlas/landmark/2d/skulls/data/skull_habilis.vtk'}],
                [{'skull': example_data_dir + '/atlas/landmark/2d/skulls/data/skull_neandertalis.vtk'}],
                [{'skull': example_data_dir + '/atlas/landmark/2d/skulls/data/skull_sapiens.vtk'}]],
            'subject_ids': ['australopithecus', 'erectus', 'habilis', 'neandertalis', 'sapiens'],
        }
        template_specifications = {
            'skull': {'deformable_object_type': 'polyline',
                      'kernel_type': 'torch', 'kernel_width': 20.0,
                      'noise_std': 1.0,
                      'filename': example_data_dir + '/atlas/landmark/2d/skulls/data/template.vtk',
                      'attachment_type': 'varifold'}}

        self.deformetrica.estimate_deterministic_atlas(
            template_specifications,
            dataset_specifications,
            estimator_options={'optimization_method_type': 'GradientAscent', 'initial_step_size': 1.,
                               'max_iterations': 10, 'max_line_search_iterations': 10,
                               'callback': self.__estimator_callback_stop},
            model_options={'deformation_kernel_type': 'torch', 'deformation_kernel_width': 40.0})

        self.assertTrue(self.has_estimator_callback_been_called)
        self.assertEqual(1, self.current_iteration)

    #
    # Deterministic Atlas
    #

    def test_estimate_deterministic_atlas_landmark_2d_skulls(self):
        dataset_specifications = {
            'dataset_filenames': [
                [{'skull': example_data_dir + '/atlas/landmark/2d/skulls/data/skull_australopithecus.vtk'}],
                [{'skull': example_data_dir + '/atlas/landmark/2d/skulls/data/skull_erectus.vtk'}],
                [{'skull': example_data_dir + '/atlas/landmark/2d/skulls/data/skull_habilis.vtk'}],
                [{'skull': example_data_dir + '/atlas/landmark/2d/skulls/data/skull_neandertalis.vtk'}],
                [{'skull': example_data_dir + '/atlas/landmark/2d/skulls/data/skull_sapiens.vtk'}]],
            'subject_ids': ['australopithecus', 'erectus', 'habilis', 'neandertalis', 'sapiens'],
        }
        template_specifications = {
            'skull': {'deformable_object_type': 'polyline',
                      'kernel_type': 'torch', 'kernel_width': 20.0,
                      'noise_std': 1.0,
                      'filename': example_data_dir + '/atlas/landmark/2d/skulls/data/template.vtk',
                      'attachment_type': 'varifold'}}

        self.deformetrica.estimate_deterministic_atlas(
            template_specifications,
            dataset_specifications,
            estimator_options={'optimization_method_type': 'GradientAscent', 'initial_step_size': 1.,
                               'max_iterations': 10, 'max_line_search_iterations': 10,
                               'callback': self.__estimator_callback},
            model_options={'deformation_kernel_type': 'torch', 'deformation_kernel_width': 40.0, 'dtype': 'float64'})

        self.assertTrue(self.has_estimator_callback_been_called)

    def test_estimate_deterministic_atlas_landmark_3d_brain_structure(self):
        dataset_specifications = {
            'dataset_filenames': [
                [{'amygdala': example_data_dir + '/atlas/landmark/3d/brain_structures/data/amygdala1.vtk',
                  'hippo': example_data_dir + '/atlas/landmark/3d/brain_structures/data/hippo1.vtk'}],
                [{'amygdala': example_data_dir + '/atlas/landmark/3d/brain_structures/data/amygdala2.vtk',
                  'hippo': example_data_dir + '/atlas/landmark/3d/brain_structures/data/hippo2.vtk'}],
                [{'amygdala': example_data_dir + '/atlas/landmark/3d/brain_structures/data/amygdala3.vtk',
                  'hippo': example_data_dir + '/atlas/landmark/3d/brain_structures/data/hippo3.vtk'}],
                [{'amygdala': example_data_dir + '/atlas/landmark/3d/brain_structures/data/amygdala4.vtk',
                  'hippo': example_data_dir + '/atlas/landmark/3d/brain_structures/data/hippo4.vtk'}]],
            'subject_ids': ['subj1', 'subj2', 'subj3', 'subj4']
        }
        template_specifications = {
            'amygdala': {'deformable_object_type': 'SurfaceMesh',
                         'kernel_type': 'torch', 'kernel_width': 15.0,
                         'noise_std': 10.0,
                         'filename': example_data_dir + '/atlas/landmark/3d/brain_structures/data/amyg_prototype.vtk',
                         'attachment_type': 'varifold'},
            'hippo': {'deformable_object_type': 'SurfaceMesh',
                      'kernel_type': 'torch', 'kernel_width': 15.0,
                      'noise_std': 6.0,
                      'filename': example_data_dir + '/atlas/landmark/3d/brain_structures/data/hippo_prototype.vtk',
                      'attachment_type': 'varifold'}
        }

        self.deformetrica.estimate_deterministic_atlas(
            template_specifications,
            dataset_specifications,
            estimator_options={'optimization_method_type': 'ScipyLBFGS', 'max_iterations': 10,
                               'callback': self.__estimator_callback},
            model_options={'deformation_kernel_type': 'torch', 'deformation_kernel_width': 7.0,
                           'freeze_template': False, 'freeze_control_points': True})

        self.assertTrue(self.has_estimator_callback_been_called)

    def test_estimate_deterministic_atlas_image_2d_digits(self):
        dataset_specifications = {
            'dataset_filenames': [[{'img': example_data_dir + '/atlas/image/2d/digits/data/digit_2_sample_1.png'}],
                                  [{'img': example_data_dir + '/atlas/image/2d/digits/data/digit_2_sample_2.png'}],
                                  [{'img': example_data_dir + '/atlas/image/2d/digits/data/digit_2_sample_3.png'}],
                                  [{'img': example_data_dir + '/atlas/image/2d/digits/data/digit_2_sample_4.png'}],
                                  [{'img': example_data_dir + '/atlas/image/2d/digits/data/digit_2_sample_5.png'}],
                                  [{'img': example_data_dir + '/atlas/image/2d/digits/data/digit_2_sample_6.png'}],
                                  [{'img': example_data_dir + '/atlas/image/2d/digits/data/digit_2_sample_7.png'}],
                                  [{'img': example_data_dir + '/atlas/image/2d/digits/data/digit_2_sample_8.png'}],
                                  [{'img': example_data_dir + '/atlas/image/2d/digits/data/digit_2_sample_9.png'}],
                                  [{'img': example_data_dir + '/atlas/image/2d/digits/data/digit_2_sample_10.png'}],
                                  [{'img': example_data_dir + '/atlas/image/2d/digits/data/digit_2_sample_11.png'}],
                                  [{'img': example_data_dir + '/atlas/image/2d/digits/data/digit_2_sample_12.png'}],
                                  [{'img': example_data_dir + '/atlas/image/2d/digits/data/digit_2_sample_13.png'}],
                                  [{'img': example_data_dir + '/atlas/image/2d/digits/data/digit_2_sample_14.png'}],
                                  [{'img': example_data_dir + '/atlas/image/2d/digits/data/digit_2_sample_15.png'}],
                                  [{'img': example_data_dir + '/atlas/image/2d/digits/data/digit_2_sample_16.png'}],
                                  [{'img': example_data_dir + '/atlas/image/2d/digits/data/digit_2_sample_17.png'}],
                                  [{'img': example_data_dir + '/atlas/image/2d/digits/data/digit_2_sample_18.png'}],
                                  [{'img': example_data_dir + '/atlas/image/2d/digits/data/digit_2_sample_19.png'}],
                                  [{'img': example_data_dir + '/atlas/image/2d/digits/data/digit_2_sample_20.png'}]],
            'subject_ids': ['sub1', 'sub2', 'sub3', 'sub4', 'sub5', 'sub6', 'sub7', 'sub8', 'sub9', 'sub10',
                            'sub11', 'sub12', 'sub13', 'sub14', 'sub15', 'sub16', 'sub17', 'sub18', 'sub19', 'sub20']
        }
        template_specifications = {
            'img': {'deformable_object_type': 'Image',
                    'noise_std': 0.1,
                    'filename': example_data_dir + '/atlas/image/2d/digits/data/digit_2_mean.png'}}

        self.deformetrica.estimate_deterministic_atlas(
            template_specifications, dataset_specifications,
            estimator_options={'optimization_method_type': 'ScipyLBFGS', 'max_iterations': 10,
                               'convergence_tolerance': 1e-5,
                               'callback': self.__estimator_callback},
            model_options={'deformation_kernel_type': 'torch', 'deformation_kernel_width': 2.0})

    #
    # Bayesian Atlas
    #

    def test_estimate_bayesian_atlas_landmark_2d_skulls(self):
        dataset_specifications = {
            'dataset_filenames': [
                [{'skull': example_data_dir + '/atlas/landmark/2d/skulls/data/skull_australopithecus.vtk'}],
                [{'skull': example_data_dir + '/atlas/landmark/2d/skulls/data/skull_erectus.vtk'}],
                [{'skull': example_data_dir + '/atlas/landmark/2d/skulls/data/skull_habilis.vtk'}],
                [{'skull': example_data_dir + '/atlas/landmark/2d/skulls/data/skull_neandertalis.vtk'}],
                [{'skull': example_data_dir + '/atlas/landmark/2d/skulls/data/skull_sapiens.vtk'}]],
            'subject_ids': ['australopithecus', 'erectus', 'habilis', 'neandertalis', 'sapiens']
        }
        template_specifications = {
            'skull': {'deformable_object_type': 'polyline',
                      'kernel_type': 'torch',
                      'kernel_width': 20.0,
                      'noise_std': 1.0,
                      'noise_variance_prior_normalized_dof': 10,
                      'noise_variance_prior_scale_std': 1,
                      'filename': example_data_dir + '/atlas/landmark/2d/skulls/data/template.vtk',
                      'attachment_type': 'varifold'}}

        self.deformetrica.estimate_bayesian_atlas(
            template_specifications, dataset_specifications,
            estimator_options={'optimization_method_type': 'GradientAscent', 'initial_step_size': 1.,
                               'max_iterations': 10, 'max_line_search_iterations': 10},
            model_options={'deformation_kernel_type': 'torch', 'deformation_kernel_width': 40.0})

    # Longitudinal Atlas

    def test_estimate_longitudinal_atlas(self):
        BASE_DIR = sandbox_data_dir + '/longitudinal_atlas/landmark/2d/starmen'

        dataset_specifications = {'dataset_filenames': [], 'visit_ages': []}

        subject_ids = []
        # visit_ages = []
        for subject_id in range(0, 5):
            subject_ids.append('s' + str(subject_id))
            subject_visits = []
            for visit_id in range(0, 5):
                file_name = 'subject_' + str(subject_id) + '__tp_' + str(visit_id) + '.vtk'
                subject_visits.append(
                    {'starman': os.path.join(BASE_DIR, 'data', file_name)})

            dataset_specifications['dataset_filenames'].append(subject_visits)
            dataset_specifications['visit_ages'].append(list(range(68, 72)))

        dataset_specifications['subject_ids'] = subject_ids

        template_specifications = {
            'starman': {'deformable_object_type': 'polyline',
                        'noise_std': 1.0,
                        'filename': os.path.join(
                            BASE_DIR, 'data', 'ForInitialization__Template_starman__FromLongitudinalAtlas.vtk'),
                        'attachment_type': 'landmark',
                        'noise_variance_prior_normalized_dof': 0.01,
                        'noise_variance_prior_scale_std': 1.}}

        start = time.perf_counter()

        self.deformetrica.estimate_longitudinal_atlas(
            template_specifications,
            dataset_specifications,
            estimator_options={'optimization_method_type': 'GradientAscent', 'initial_step_size': 1e-5,
                               'max_iterations': 10, 'max_line_search_iterations': 10},
            model_options={'deformation_kernel_type': 'torch', 'deformation_kernel_width': 1.0,
                           'initial_control_points': os.path.join(
                               BASE_DIR, 'data', 'ForInitialization__ControlPoints__FromLongitudinalAtlas.txt'),
                           'initial_momenta': os.path.join(
                               BASE_DIR, 'data', 'ForInitialization__Momenta__FromLongitudinalAtlas.txt'),
                           't0': 72.2,
                           'initial_modulation_matrix': os.path.join(
                               BASE_DIR, 'data', 'ForInitialization__ModulationMatrix__FromLongitudinalAtlas.txt'),
                           'initial_sources': os.path.join(
                               BASE_DIR, 'data', 'ForInitialization__Sources__FromLongitudinalAtlas.txt'),
                           'initial_accelerations': os.path.join(
                               BASE_DIR, 'data', 'ForInitialization__Accelerations__FromLongitudinalAtlas.txt'),
                           'initial_onset_ages': os.path.join(
                               BASE_DIR, 'data', 'ForInitialization__OnsetAges__FromLongitudinalAtlas.txt'),
                           'number_of_threads': 2})
        print('>>>>> estimate_longitudinal_atlas took : ' + str(time.perf_counter() - start) + ' seconds')

    @unittest.skip
    def test_estimate_longitudinal_atlas_hippocampi(self):
        import torch
        import numpy as np
        torch.manual_seed(42)
        np.random.seed(42)

        BASE_DIR = sandbox_data_dir + '/longitudinal_atlas/landmark/3d/hippocampi'

        dataset_specifications = {'dataset_filenames': [], 'visit_ages': []}

        subject_ids = [
            '002S0729', '002S0954', '002S1070', '002S1268', '002S4171', '002S4262',
            '002S4521', '003S1057', '005S0222', '005S0223', '005S0448', '005S0572',
            # '005S1224', '006S0675', '006S1130', '006S4346', '006S4363', '006S4515',
            # '007S0041', '007S0101', '007S0128', '007S0249', '007S0293', '007S0344',
            # '007S0698', '007S2106', '009S1030', '009S2381', '009S4324', '009S4530',
            # '009S4958', '010S0161', '010S0904', '011S0241', '011S0326', '011S0362',
            # '011S0856', '011S0861', '011S1080', '011S1282', '011S4366', '011S4893',
            # '012S1033', '012S1292', '012S4094', '012S4188', '012S5121', '013S0240',
            # '013S0325', '013S0860', '013S1186', '013S4595', '014S0548', '014S0558',
            # '014S0563', '014S0658', '014S4058', '014S4079', '014S4263', '014S4668',
            # '016S0769', '016S1117', '016S1121', '016S1138', '016S1326', '016S4584',
            # '016S4902', '016S5031', '018S0057', '018S0080', '018S0142', '018S0142',
            # '018S0155', '018S0406', '018S0450', '018S4597', '019S4293', '019S4680',
            # '021S0141', '021S0231', '021S0276', '021S0424', '021S0626', '021S0984',
            # '021S4245', '021S4402', '021S4659', '021S4857', '022S0750', '022S1097',
            # '022S1351', '022S1394', '022S2087', '023S0030', '023S0042', '023S0061',
            # '023S0126', '023S0217', '023S0331', '023S0376', '023S0388', '023S0604',
            # '023S0625', '023S0855', '023S0887', '023S1126', '023S1247', '023S4035',
            # '023S4243', '023S4502', '023S4796', '024S0985', '024S1393', '027S0179',
            # '027S0256', '027S0408', '027S0461', '027S0835', '027S1213', '027S1387',
            # '027S4729', '027S4757', '027S4936', '027S4943', '029S0843', '029S0878',
            # '029S0914', '029S1073', '029S1318', '029S4385', '031S0294', '031S0568',
            # '031S0830', '031S1066', '031S4042', '031S4203', '032S0187', '032S0214',
            # '032S0978', '032S4823', '033S0511', '033S0513', '033S0514', '033S0567',
            # '033S0723', '033S0725', '033S0906', '033S0920', '033S0922', '033S1098',
            # '033S1116', '035S0204', '035S0292', '035S0997', '035S4114', '035S4414',
            # '035S4582', '035S4784', '036S0869', '036S0945', '036S0976', '036S1135',
            # '036S1240', '036S4714', '036S4715', '037S0467', '037S0501', '037S0539',
            # '037S0552', '037S0566', '037S0588', '037S1078', '037S1225', '037S4015',
            # '037S4030', '037S4432', '041S0314', '041S0549', '041S0898', '041S1010',
            # '041S1412', '041S1423', '041S1425', '041S4041', '041S4720', '041S5026',
            # '051S1123', '051S1331', '051S4929', '052S0671', '052S0952', '052S1054',
            # '052S1346', '052S4807', '052S4945', '053S0389', '053S0507', '053S4661',
            # '057S0839', '057S0941', '057S1007', '057S1217', '057S1265', '057S2398',
            # '057S4888', '062S1299', '067S0045', '067S0077', '067S0098', '067S0243',
            # '067S0336', '067S2195', '067S4918', '068S0442', '068S0476', '068S0872',
            # '068S2248', '068S2316', '072S4057', '072S4102', '072S4131', '072S4462',
            # '073S0518', '073S0909', '073S4777', '082S0832', '094S0434', '094S1015',
            # '094S1398', '094S2216', '094S4162', '098S0160', '098S0269', '098S0667',
            # '098S2047', '098S4506', '099S0051', '099S0054', '099S0111', '099S4157',
            # '100S0892', '100S0930', '114S0378', '114S1106', '116S0361', '116S0649',
            # '116S0752', '116S0834', '116S1243', '116S1249', '116S1271', '116S1315',
            # '116S4167', '121S1350', '123S0050', '123S0106', '123S0108', '123S0390',
            # '123S4096', '126S0708', '126S0865', '126S1077', '126S4458', '126S4507',
            # '126S4675', '126S4712', '127S0259', '127S0394', '127S0925', '127S1032',
            # '127S1427', '127S2213', '127S4240', '127S4765', '127S4844', '127S4928',
            # '128S0227', '128S0230', '128S0258', '128S0611', '128S0947', '128S1043',
            # '128S1148', '128S1406', '128S1407', '128S2130', '129S0778', '130S0285',
            # '130S0289', '130S0423', '130S2403', '130S4250', '130S4415', '130S4542',
            # '131S0123', '131S1389', '132S0987', '133S0638', '133S0727', '133S0913',
            # '135S4406', '135S4689', '136S0195', '136S0695', '136S0873', '136S0874',
            # '136S4189', '137S0631', '137S0972', '137S0973', '137S0994', '137S4303',
            # '137S4331', '137S4596', '137S4623', '137S4631', '137S4815', '137S4816',
            # '141S0697', '141S0915', '141S0982', '141S1004', '141S1244', '141S1255',
            # '941S1295', '941S1311'
        ]
        # visit_ages = []
        for subject_id in subject_ids:
            subject_visits = []
            subject_visit_ages = []
            for i in ["%02d" % x for x in range(110)]:
                file_name = 'sub-ADNI' + str(subject_id) + '_ses-M' + str(i) + '.vtk'

                if os.path.isfile(os.path.join(BASE_DIR, 'data', file_name)):   # only add if file exists
                    subject_id, visit_age = adni_extract_from_file_name(file_name)
                    subject_visit_ages.append(float(visit_age))
                    subject_visits.append({'hippocampi': os.path.join(BASE_DIR, 'data', file_name)})

            assert len(subject_visits) > 0 or len(subject_visit_ages) > 0, \
                'len(subject_visits)=' + str(len(subject_visits)) + ', ' \
                'len(subject_visit_ages)=' + str(len(subject_visit_ages)) + ' does the subject exist ?'
            dataset_specifications['dataset_filenames'].append(subject_visits)
            dataset_specifications['visit_ages'].append(subject_visit_ages)

        dataset_specifications['subject_ids'] = subject_ids

        template_specifications = {
            'hippocampi': {'deformable_object_type': 'SurfaceMesh',
                           'noise_std': 5.0,
                           'kernel_type': 'torch',
                           'kernel_width': 5.0,
                           'filename': os.path.join(BASE_DIR, 'data', 'ForInitialization_Template_FromRegression_Smooth.vtk'),
                           'attachment_type': 'current',
                           'noise_variance_prior_normalized_dof': 0.01,
                           'noise_variance_prior_scale_std': 1.}
        }

        start = time.perf_counter()

        self.deformetrica.estimate_longitudinal_atlas(
            template_specifications,
            dataset_specifications,
            estimator_options={'optimization_method_type': 'McmcSaem', 'initial_step_size': 1e-8,
                               'max_iterations': 2, 'max_line_search_iterations': 5, 'sample_every_n_mcmc_iters': 10,
                               'use_sobolev_gradient': True},
            model_options={'deformation_kernel_type': 'keops', 'deformation_kernel_width': 10.0,
                           'concentration_of_timepoints': 5, 'number_of_timepoints': 6,
                           'initial_control_points': os.path.join(BASE_DIR, 'data', 'ForInitialization_ControlPoints_FromRegression_s0671_tp27.txt'),
                           'initial_momenta': os.path.join(BASE_DIR, 'data', 'ForInitialization_Momenta_FromRegression_s0671_tp27.txt'),
                           'initial_modulation_matrix': os.path.join(BASE_DIR, 'data', 'ForInitialization_ModulationMatrix_FromAtlas.txt'),
                           'number_of_threads': 6})
        print('>>>>> estimate_longitudinal_atlas took : ' + str(time.perf_counter() - start) + ' seconds')

    #
    # Affine Atlas
    #

    def test_estimate_affine_atlas(self):
        dataset_specifications = {
            'dataset_filenames': [
                [{'amygdala': example_data_dir + '/atlas/landmark/3d/brain_structures/data/amygdala1.vtk'}],
                [{'amygdala': example_data_dir + '/atlas/landmark/3d/brain_structures/data/amygdala2.vtk'}],
                [{'amygdala': example_data_dir + '/atlas/landmark/3d/brain_structures/data/amygdala3.vtk'}],
                [{'amygdala': example_data_dir + '/atlas/landmark/3d/brain_structures/data/amygdala4.vtk'}]],
            'subject_ids': ['subj1', 'subj2', 'subj3', 'subj4'],
            'visit_ages': [[1], [6], [6], [4]]
        }
        template_specifications = {
            'amygdala': {'deformable_object_type': 'SurfaceMesh',
                         'kernel_type': 'torch', 'kernel_width': 5.0,
                         'noise_std': 10.0,
                         'filename': example_data_dir + '/atlas/landmark/3d/brain_structures/data/amyg_prototype.vtk',
                         'attachment_type': 'current'}
        }

        self.deformetrica.estimate_affine_atlas(template_specifications, dataset_specifications,
                                                estimator_options={'optimization_method_type': 'GradientAscent', 'initial_step_size': 1.,
                                                                   'max_iterations': 4, 'max_line_search_iterations': 10},
                                                model_options={'deformation_kernel_type': 'torch', 'deformation_kernel_width': 40.0})

    #
    # Regression
    #

    def test_estimate_geodesic_regression_landmark_2d_skulls(self):
        dataset_specifications = {
            'dataset_filenames': [
                [{'skull': example_data_dir + '/regression/landmark/2d/skulls/data/skull_australopithecus.vtk'},
                 {'skull': example_data_dir + '/regression/landmark/2d/skulls/data/skull_habilis.vtk'},
                 {'skull': example_data_dir + '/regression/landmark/2d/skulls/data/skull_erectus.vtk'},
                 {'skull': example_data_dir + '/regression/landmark/2d/skulls/data/skull_sapiens.vtk'}]],
            'visit_ages': [[1, 2, 3, 4]],
            'subject_ids': [['australopithecus', 'habilis', 'erectus', 'sapiens']]
        }
        template_specifications = {
            'skull': {'deformable_object_type': 'polyline',
                      'kernel_type': 'torch', 'kernel_width': 20.0,
                      'noise_std': 1.0,
                      'filename': example_data_dir + '/regression/landmark/2d/skulls/data/template.vtk',
                      'attachment_type': 'varifold'}}

        self.deformetrica.estimate_geodesic_regression(
            template_specifications, dataset_specifications,
            estimator_options={'optimization_method_type': 'GradientAscent', 'max_iterations': 10},
            model_options={'deformation_kernel_type': 'torch', 'deformation_kernel_width': 25.0,
                           'concentration_of_time_points': 5, 'smoothing_kernel_width': 20.0})

    def test_estimate_geodesic_regression_landmark_3d_surprise(self):
        dataset_specifications = {
            'dataset_filenames': [
                [{'skull': example_data_dir + '/regression/landmark/3d/surprise/data/sub-F001_ses-000.vtk'},
                 {'skull': example_data_dir + '/regression/landmark/3d/surprise/data/sub-F001_ses-005.vtk'},
                 {'skull': example_data_dir + '/regression/landmark/3d/surprise/data/sub-F001_ses-010.vtk'},
                 {'skull': example_data_dir + '/regression/landmark/3d/surprise/data/sub-F001_ses-015.vtk'},
                 {'skull': example_data_dir + '/regression/landmark/3d/surprise/data/sub-F001_ses-020.vtk'},
                 {'skull': example_data_dir + '/regression/landmark/3d/surprise/data/sub-F001_ses-025.vtk'},
                 {'skull': example_data_dir + '/regression/landmark/3d/surprise/data/sub-F001_ses-030.vtk'},
                 {'skull': example_data_dir + '/regression/landmark/3d/surprise/data/sub-F001_ses-035.vtk'}]],
            'visit_ages': [[0, 5, 10, 15, 20, 25, 30, 35]],
            'subject_ids': [['ses-000', 'ses-005', 'ses-010', 'ses-015', 'ses-020', 'ses-025', 'ses-030', 'ses-035']]
        }
        template_specifications = {
            'skull': {'deformable_object_type': 'polyline',
                      'noise_std': 0.0035,
                      'filename': example_data_dir + '/regression/landmark/3d/surprise/data/ForInitialization__Template__FromUser.vtk',
                      'attachment_type': 'landmark'}}

        self.deformetrica.estimate_geodesic_regression(
            template_specifications, dataset_specifications,
            estimator_options={'optimization_method_type': 'GradientAscent', 'max_iterations': 10,
                               'convergence_tolerance': 1e-5, 'initial_step_size': 1e-6},
            model_options={'deformation_kernel_type': 'torch', 'deformation_kernel_width': 0.015,
                           'concentration_of_time_points': 1, 'smoothing_kernel_width': 20.0, 't0': 5.5,
                           'use_sobolev_gradient': True, 'dense_mode': True})

    def test_estimate_geodesic_regression_image_2d_cross(self):
        dataset_specifications = {
            'dataset_filenames': [[{'skull': example_data_dir + '/regression/image/2d/cross/data/cross_-5.png'},
                                   {'skull': example_data_dir + '/regression/image/2d/cross/data/cross_-3.png'},
                                   {'skull': example_data_dir + '/regression/image/2d/cross/data/cross_-2.png'},
                                   {'skull': example_data_dir + '/regression/image/2d/cross/data/cross_0.png'},
                                   {'skull': example_data_dir + '/regression/image/2d/cross/data/cross_1.png'},
                                   {'skull': example_data_dir + '/regression/image/2d/cross/data/cross_3.png'}]],
            'visit_ages': [[-5, -3, -2, 0, 1, 3]],
            'subject_ids': [['t-5', 't-3', 't-2', 't0', 't1', 't3']]
        }
        template_specifications = {
            'skull': {'deformable_object_type': 'image',
                      'noise_std': 0.1,
                      'filename': example_data_dir + '/regression/image/2d/cross/data/cross_0.png',
                      'attachment_type': 'varifold'}}

        self.deformetrica.estimate_geodesic_regression(
            template_specifications, dataset_specifications,
            estimator_options={'optimization_method_type': 'GradientAscent', 'max_iterations': 10,
                               'initial_step_size': 1e-9},
            model_options={'deformation_kernel_type': 'torch', 'deformation_kernel_width': 10.0,
                           'concentration_of_time_points': 3, 'freeze_template': True})

    #
    # Registration
    #

    def test_estimate_deterministic_registration_landmark_2d_points(self):
        dataset_specifications = {
            'dataset_filenames': [
                [{'pointcloud': example_data_dir + '/registration/landmark/2d/points/data/target_points.vtk'}]],
            'subject_ids': ['target']
        }
        template_specifications = {
            'pointcloud': {'deformable_object_type': 'landmark',
                           'noise_std': 1e-3,
                           'filename': example_data_dir + '/registration/landmark/2d/points/data/source_points.vtk'}}

        self.deformetrica.estimate_deterministic_atlas(
            template_specifications, dataset_specifications,
            estimator_options={'optimization_method_type': 'GradientAscent', 'initial_step_size': 1e-8,
                               'max_iterations': 10, 'max_line_search_iterations': 200},
            model_options={'deformation_kernel_type': 'torch', 'deformation_kernel_width': 3.0,
                           'number_of_time_points': 10, 'freeze_template': True, 'freeze_control_points': True})

    def test_estimate_deterministic_registration_landmark_2d_starfish(self):
        dataset_specifications = {
            'dataset_filenames': [
                [{'starfish': example_data_dir + '/registration/landmark/2d/starfish/data/starfish_target.vtk'}]],
            'subject_ids': ['target']
        }
        template_specifications = {
            'starfish': {'deformable_object_type': 'polyline',
                         'kernel_type': 'torch', 'kernel_width': 50.0,
                         'noise_std': 0.1,
                         'attachment_type': 'current',
                         'filename': example_data_dir + '/registration/landmark/2d/starfish/data/starfish_reference.vtk'}}

        self.deformetrica.estimate_deterministic_atlas(
            template_specifications, dataset_specifications,
            estimator_options={'optimization_method_type': 'ScipyLBFGS', 'max_iterations': 10},
            model_options={'deformation_kernel_type': 'torch', 'deformation_kernel_width': 30.0,
                           'number_of_time_points': 10, 'freeze_template': True, 'freeze_control_points': True})

    def test_estimate_deterministic_registration_image_2d_tetris(self):
        dataset_specifications = {
            'dataset_filenames': [
                [{'image': example_data_dir + '/registration/image/2d/tetris/data/image2.png'}]],
            'subject_ids': ['target']
        }
        template_specifications = {
            'image': {'deformable_object_type': 'image',
                      'kernel_type': 'torch',
                      'kernel_width': 10.0,
                      'noise_std': 0.1,
                      'filename': example_data_dir + '/registration/image/2d/tetris/data/image1.png'}}

        self.deformetrica.estimate_deterministic_atlas(
            template_specifications, dataset_specifications,
            estimator_options={'optimization_method_type': 'GradientAscent', 'max_iterations': 10},
            model_options={'deformation_kernel_type': 'torch', 'deformation_kernel_width': 20.0})

    #
    # Parallel Transport
    #

    def test_compute_parallel_transport_image_2d_snowman(self):
        BASE_DIR = example_data_dir + '/parallel_transport/image/2d/snowman/'
        template_specifications = {
            'image': {'deformable_object_type': 'image',
                      'noise_std': 0.05,
                      'filename': BASE_DIR + 'data/I1.png'}}
        self.deformetrica.compute_parallel_transport(
            template_specifications,
            model_options={'deformation_kernel_type': 'torch', 'deformation_kernel_width': 15.0,
                           'initial_control_points': BASE_DIR + 'data/Reference_progression_ControlPoints.txt',
                           'initial_momenta': BASE_DIR + 'data/Reference_progression_Momenta.txt',
                           'initial_control_points_to_transport': BASE_DIR + 'data/Registration_ControlPoints.txt',
                           'initial_momenta_to_transport': BASE_DIR + 'data/Registration_Momenta.txt',
                           'tmin': 0, 'tmax': 1, 'concentration_of_time_points': 10})

    @unittest.skipIf(not torch.cuda.is_available(), 'cuda is not available')
    @unittest.skipIf(platform in ['darwin'], 'keops kernel not available')
    def test_compute_parallel_transport_mesh_3d_alien(self):
        BASE_DIR = functional_tests_data_dir + '/parallel_transport/alien/'
        template_specifications = {
            'mesh': {'deformable_object_type': 'SurfaceMesh',
                     'filename': BASE_DIR + 'data/face.vtk',
                     'attachment_type': 'Landmark',
                     'noise_std': 1.}}
        self.deformetrica.compute_parallel_transport(
            template_specifications,
            model_options={'deformation_kernel_type': 'keops', 'deformation_kernel_width': 0.005,
                           'initial_control_points': BASE_DIR + 'data/control_points.txt',
                           'initial_momenta': BASE_DIR + 'data/momenta.txt',
                           'initial_momenta_to_transport': BASE_DIR + 'data/momenta_to_transport.txt',
                           'tmin': 0, 'tmax': 9, 'concentration_of_time_points': 3})

    #
    # Shooting
    #

    def test_compute_shooting_image_2d_snowman(self):
        BASE_DIR = example_data_dir + '/shooting/image/2d/snowman/'
        template_specifications = {
            'image': {'deformable_object_type': 'image',
                      'noise_std': 0.05,
                      'filename': BASE_DIR + 'data/I1.png'}}

        self.deformetrica.compute_shooting(
            template_specifications,
            model_options={'deformation_kernel_type': 'torch', 'deformation_kernel_width': 35.0,
                           'initial_control_points': BASE_DIR + 'data/control_points.txt',
                           'initial_momenta': BASE_DIR + 'data/momenta.txt'})

    def test_compute_shooting_image_2d_snowman_with_different_shoot_kernels(self):
        BASE_DIR = example_data_dir + '/shooting/image/2d/snowman/'
        template_specifications = {
            'image': {'deformable_object_type': 'image',
                      'noise_std': 0.05,
                      'filename': BASE_DIR + 'data/I1.png'}}

        self.deformetrica.compute_shooting(
            template_specifications,
            model_options={'deformation_kernel_type': 'torch', 'deformation_kernel_width': 35.0,
                           'shoot_kernel_type': 'torch',
                           'initial_control_points': BASE_DIR + 'data/control_points.txt',
                           'initial_momenta': BASE_DIR + 'data/momenta.txt'})
