import logging
import os
import unittest

from api.deformetrica import Deformetrica
from unit_tests import example_data_dir, sandbox_data_dir

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
            model_options={'deformation_kernel_type': 'torch', 'deformation_kernel_width': 40.0})

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

    # def test_estimate_longitudinal_atlas(self):
    #
    #     dataset_specifications = {'dataset_filenames': [], 'visit_ages': []}
    #
    #     subject_ids = []
    #     # visit_ages = []
    #     for subject_id in range(0, 5):
    #         subject_ids.append('s' + str(subject_id))
    #         subject_visits = []
    #         for visit_id in range(0, 5):
    #             file_name = 'subject_' + str(subject_id) + '__tp_' + str(visit_id) + '.vtk'
    #             subject_visits.append({'starman': sandbox_data_dir + '/longitudinal_atlas/landmark/2d/starmen/data/' + file_name})
    #             # subject_visits.append([sandbox_data_dir + '/longitudinal_atlas/landmark/2d/starmen/data/' + file_name])
    #
    #         dataset_specifications['dataset_filenames'].append(subject_visits)
    #         dataset_specifications['visit_ages'].append(list(range(68, 72)))
    #
    #     dataset_specifications['subject_ids'] = subject_ids
    #
    #     template_specifications = {
    #         'starman': {'deformable_object_type': 'polyline',
    #                     'noise_std': 1.0,
    #                     'filename': sandbox_data_dir + '/longitudinal_atlas/landmark/2d/starmen/data/ForInitialization_Template.vtk',
    #                     'attachment_type': 'landmark',
    #                     'noise_variance_prior_normalized_dof': 0.01,
    #                     'noise_variance_prior_scale_std': 1.}}
    #
    #     self.deformetrica.estimate_longitudinal_atlas(template_specifications, dataset_specifications,
    #                                                   estimator_options={'optimization_method_type': 'GradientAscent', 'initial_step_size': 1.,
    #                                                                      'max_iterations': 4, 'max_line_search_iterations': 10},
    #                                                   model_options={'deformation_kernel_type': 'torch', 'deformation_kernel_width': 40.0})

    #
    # Affine Atlas
    #

    def test_estimate_affine_atlas_aorta(self):
        dataset_specifications = {
            'dataset_filenames': [
                [{'amygdala': example_data_dir + '/atlas/landmark/3d/brain_structures/data/amygdala1.vtk'}],
                [{'amygdala': example_data_dir + '/atlas/landmark/3d/brain_structures/data/amygdala2.vtk'}],
                [{'amygdala': example_data_dir + '/atlas/landmark/3d/brain_structures/data/amygdala3.vtk'}],
                [{'amygdala': example_data_dir + '/atlas/landmark/3d/brain_structures/data/amygdala4.vtk'}]],
            'subject_ids': ['subj1', 'subj2', 'subj3', 'subj4'],
            'visit_ages': []
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
            estimator_options={'optimization_method_type': 'GradientAscent', 'max_iterations': 100},
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
            estimator_options={'optimization_method_type': 'GradientAscent', 'max_iterations': 50,
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
            estimator_options={'optimization_method_type': 'GradientAscent', 'max_iterations': 50,
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
                               'max_iterations': 100, 'max_line_search_iterations': 200},
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
            estimator_options={'optimization_method_type': 'ScipyLBFGS', 'max_iterations': 200},
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
            estimator_options={'optimization_method_type': 'GradientAscent'},
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
