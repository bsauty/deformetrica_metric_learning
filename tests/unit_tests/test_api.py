import os
import unittest
import logging

import torch

import support.kernels as kernel_factory
from api.deformetrica import Deformetrica
from core.estimators.gradient_ascent import GradientAscent
from core.estimators.scipy_optimize import ScipyOptimize
from in_out.dataset_functions import create_dataset

logging.basicConfig(level=logging.DEBUG)


class API(unittest.TestCase):

    def setUp(self):
        self.deformetrica = Deformetrica(output_dir=os.path.join(os.path.dirname(__file__), 'output'))
        self.has_estimator_callback_been_called = False
        self.current_iteration = 0

    def __estimator_callback(self, status_dict):
        print(status_dict)
        self.assertTrue('current_iteration' in status_dict)
        self.assertTrue('current_log_likelihood' in status_dict)
        self.assertTrue('current_attachment' in status_dict)
        self.assertTrue('current_regularity' in status_dict)
        self.current_iteration = status_dict['current_iteration']
        self.has_estimator_callback_been_called = True
        return True

    def __estimator_callback_stop(self, status_dict):
        self.__estimator_callback(status_dict)
        return False

    def test_estimator_loop_stop(self):
        dataset_file_names = [[{'skull': '../../examples/atlas/landmark/2d/skulls/data/skull_australopithecus.vtk'}],
                              [{'skull': '../../examples/atlas/landmark/2d/skulls/data/skull_erectus.vtk'}],
                              [{'skull': '../../examples/atlas/landmark/2d/skulls/data/skull_habilis.vtk'}],
                              [{'skull': '../../examples/atlas/landmark/2d/skulls/data/skull_neandertalis.vtk'}],
                              [{'skull': '../../examples/atlas/landmark/2d/skulls/data/skull_sapiens.vtk'}]]
        visit_ages = []
        subject_ids = ['australopithecus', 'erectus', 'habilis', 'neandertalis', 'sapiens']
        template_specifications = {
            'skull': {'deformable_object_type': 'polyline',
                      'kernel': kernel_factory.factory(kernel_factory.Type.TORCH, kernel_width=20.0),
                      'noise_std': 1.0,
                      'filename': '../../examples/atlas/landmark/2d/skulls/data/template.vtk',
                      'attachment_type': 'varifold'}}

        dataset = create_dataset(dataset_file_names, visit_ages, subject_ids, template_specifications, dimension=2, tensor_scalar_type=torch.DoubleTensor)

        self.deformetrica.estimate_deterministic_atlas(template_specifications, dataset,
                                                       estimator=ScipyOptimize,
                                                       estimator_options={'initial_step_size': 1., 'max_iterations': 10, 'max_line_search_iterations': 10, 'callback': self.__estimator_callback_stop},
                                                       deformation_kernel=kernel_factory.factory(kernel_factory.Type.TORCH, kernel_width=40.0))

        self.assertTrue(self.has_estimator_callback_been_called)
        self.assertEqual(1, self.current_iteration)

    # Deterministic Atlas

    def test_estimate_deterministic_atlas_landmark_2d_skulls(self):
        dataset_file_names = [[{'skull': '../../examples/atlas/landmark/2d/skulls/data/skull_australopithecus.vtk'}],
                              [{'skull': '../../examples/atlas/landmark/2d/skulls/data/skull_erectus.vtk'}],
                              [{'skull': '../../examples/atlas/landmark/2d/skulls/data/skull_habilis.vtk'}],
                              [{'skull': '../../examples/atlas/landmark/2d/skulls/data/skull_neandertalis.vtk'}],
                              [{'skull': '../../examples/atlas/landmark/2d/skulls/data/skull_sapiens.vtk'}]]
        visit_ages = []
        subject_ids = ['australopithecus', 'erectus', 'habilis', 'neandertalis', 'sapiens']
        template_specifications = {
            'skull': {'deformable_object_type': 'polyline',
                      'kernel': kernel_factory.factory(kernel_factory.Type.TORCH, kernel_width=20.0),
                      'noise_std': 1.0,
                      'filename': '../../examples/atlas/landmark/2d/skulls/data/template.vtk',
                      'attachment_type': 'varifold'}}

        dataset = create_dataset(dataset_file_names, visit_ages, subject_ids, template_specifications, dimension=2, tensor_scalar_type=torch.DoubleTensor)

        self.deformetrica.estimate_deterministic_atlas(template_specifications, dataset,
                                                       estimator=GradientAscent,
                                                       estimator_options={'initial_step_size': 1., 'max_iterations': 10, 'max_line_search_iterations': 10, 'callback': self.__estimator_callback},
                                                       deformation_kernel=kernel_factory.factory(kernel_factory.Type.TORCH, kernel_width=40.0))

        self.assertTrue(self.has_estimator_callback_been_called)

    def test_estimate_deterministic_atlas_landmark_3d_brain_structure(self):
        dataset_file_names = [
            [{'amygdala': '../../examples/atlas/landmark/3d/brain_structures/data/amygdala1.vtk',
              'hippo': '../../examples/atlas/landmark/3d/brain_structures/data/hippo1.vtk'}],
            [{'amygdala': '../../examples/atlas/landmark/3d/brain_structures/data/amygdala2.vtk',
              'hippo': '../../examples/atlas/landmark/3d/brain_structures/data/hippo2.vtk'}],
            [{'amygdala': '../../examples/atlas/landmark/3d/brain_structures/data/amygdala3.vtk',
              'hippo': '../../examples/atlas/landmark/3d/brain_structures/data/hippo3.vtk'}],
            [{'amygdala': '../../examples/atlas/landmark/3d/brain_structures/data/amygdala4.vtk',
              'hippo': '../../examples/atlas/landmark/3d/brain_structures/data/hippo4.vtk'}]
        ]
        visit_ages = []
        subject_ids = ['subj1', 'subj2', 'subj3', 'subj4']
        template_specifications = {
            'amygdala': {'deformable_object_type': 'SurfaceMesh',
                         'kernel': kernel_factory.factory(kernel_factory.Type.TORCH, kernel_width=15.0),
                         'noise_std': 10.0,
                         'filename': '../../examples/atlas/landmark/3d/brain_structures/data/amyg_prototype.vtk',
                         'attachment_type': 'varifold'},
            'hippo': {'deformable_object_type': 'SurfaceMesh',
                      'kernel': kernel_factory.factory(kernel_factory.Type.TORCH, kernel_width=15.0),
                      'noise_std': 6.0,
                      'filename': '../../examples/atlas/landmark/3d/brain_structures/data/hippo_prototype.vtk',
                      'attachment_type': 'varifold'}
        }

        dataset = create_dataset(dataset_file_names, visit_ages, subject_ids, template_specifications, dimension=3)

        self.deformetrica.estimate_deterministic_atlas(template_specifications, dataset,
                                                       estimator=ScipyOptimize,
                                                       estimator_options={'max_iterations': 10, 'callback': self.__estimator_callback},
                                                       deformation_kernel=kernel_factory.factory(kernel_factory.Type.TORCH, kernel_width=7.0),
                                                       freeze_template=False, freeze_control_points=True)

        self.assertTrue(self.has_estimator_callback_been_called)

    def test_estimate_deterministic_atlas_image_2d_digits(self):
        dataset_file_names = [[{'img': '../../examples/atlas/image/2d/digits/data/digit_2_sample_1.png'}],
                              [{'img': '../../examples/atlas/image/2d/digits/data/digit_2_sample_2.png'}],
                              [{'img': '../../examples/atlas/image/2d/digits/data/digit_2_sample_3.png'}],
                              [{'img': '../../examples/atlas/image/2d/digits/data/digit_2_sample_4.png'}],
                              [{'img': '../../examples/atlas/image/2d/digits/data/digit_2_sample_5.png'}],
                              [{'img': '../../examples/atlas/image/2d/digits/data/digit_2_sample_6.png'}],
                              [{'img': '../../examples/atlas/image/2d/digits/data/digit_2_sample_7.png'}],
                              [{'img': '../../examples/atlas/image/2d/digits/data/digit_2_sample_8.png'}],
                              [{'img': '../../examples/atlas/image/2d/digits/data/digit_2_sample_9.png'}],
                              [{'img': '../../examples/atlas/image/2d/digits/data/digit_2_sample_10.png'}],
                              [{'img': '../../examples/atlas/image/2d/digits/data/digit_2_sample_11.png'}],
                              [{'img': '../../examples/atlas/image/2d/digits/data/digit_2_sample_12.png'}],
                              [{'img': '../../examples/atlas/image/2d/digits/data/digit_2_sample_13.png'}],
                              [{'img': '../../examples/atlas/image/2d/digits/data/digit_2_sample_14.png'}],
                              [{'img': '../../examples/atlas/image/2d/digits/data/digit_2_sample_15.png'}],
                              [{'img': '../../examples/atlas/image/2d/digits/data/digit_2_sample_16.png'}],
                              [{'img': '../../examples/atlas/image/2d/digits/data/digit_2_sample_17.png'}],
                              [{'img': '../../examples/atlas/image/2d/digits/data/digit_2_sample_18.png'}],
                              [{'img': '../../examples/atlas/image/2d/digits/data/digit_2_sample_19.png'}],
                              [{'img': '../../examples/atlas/image/2d/digits/data/digit_2_sample_20.png'}]]
        visit_ages = []
        subject_ids = ['sub1', 'sub2', 'sub3', 'sub4', 'sub5', 'sub6', 'sub7', 'sub8', 'sub9', 'sub10',
                       'sub11', 'sub12', 'sub13', 'sub14', 'sub15', 'sub16', 'sub17', 'sub18', 'sub19', 'sub20']
        template_specifications = {
            'img': {'deformable_object_type': 'Image',
                    'noise_std': 0.1,
                    'filename': '../../examples/atlas/image/2d/digits/data/digit_2_mean.png'}}

        dataset = create_dataset(dataset_file_names, visit_ages, subject_ids, template_specifications, dimension=2)

        self.deformetrica.estimate_deterministic_atlas(template_specifications, dataset,
                                                       estimator=ScipyOptimize,
                                                       estimator_options={'max_iterations': 10, 'convergence_tolerance': 1e-5},
                                                       deformation_kernel=kernel_factory.factory(kernel_factory.Type.TORCH, kernel_width=2.0))

    # Bayesian Atlas

    def test_estimate_bayesian_atlas_landmark_2d_skulls(self):
        dataset_file_names = [[{'skull': '../../examples/atlas/landmark/2d/skulls/data/skull_australopithecus.vtk'}],
                              [{'skull': '../../examples/atlas/landmark/2d/skulls/data/skull_erectus.vtk'}],
                              [{'skull': '../../examples/atlas/landmark/2d/skulls/data/skull_habilis.vtk'}],
                              [{'skull': '../../examples/atlas/landmark/2d/skulls/data/skull_neandertalis.vtk'}],
                              [{'skull': '../../examples/atlas/landmark/2d/skulls/data/skull_sapiens.vtk'}]]
        visit_ages = []
        subject_ids = ['australopithecus', 'erectus', 'habilis', 'neandertalis', 'sapiens']
        template_specifications = {
            'skull': {'deformable_object_type': 'polyline',
                      'kernel': kernel_factory.factory(kernel_factory.Type.TORCH, kernel_width=20.0),
                      'noise_std': 1.0,
                      'noise_variance_prior_normalized_dof': 10,
                      'noise_variance_prior_scale_std': 1,
                      'filename': '../../examples/atlas/landmark/2d/skulls/data/template.vtk',
                      'attachment_type': 'varifold'}}

        dataset = create_dataset(dataset_file_names, visit_ages, subject_ids, template_specifications, dimension=2, tensor_scalar_type=torch.DoubleTensor)

        self.deformetrica.estimate_bayesian_atlas(template_specifications, dataset,
                                                  estimator=GradientAscent,
                                                  estimator_options={'initial_step_size': 1., 'max_iterations': 10, 'max_line_search_iterations': 10},
                                                  deformation_kernel=kernel_factory.factory(kernel_factory.Type.TORCH, kernel_width=40.0))

    # Regression

    def test_estimate_geodesic_regression_landmark_2d_skulls(self):
        dataset_file_names = [[{'skull': '../../examples/regression/landmark/2d/skulls/data/skull_australopithecus.vtk'},
                              {'skull': '../../examples/regression/landmark/2d/skulls/data/skull_habilis.vtk'},
                              {'skull': '../../examples/regression/landmark/2d/skulls/data/skull_erectus.vtk'},
                              {'skull': '../../examples/regression/landmark/2d/skulls/data/skull_sapiens.vtk'}]]
        visit_ages = [[1, 2, 3, 4]]
        subject_ids = [['australopithecus', 'habilis', 'erectus', 'sapiens']]
        template_specifications = {
            'skull': {'deformable_object_type': 'polyline',
                      'kernel': kernel_factory.factory(kernel_factory.Type.TORCH, kernel_width=20.0),
                      'noise_std': 1.0,
                      'filename': '../../examples/regression/landmark/2d/skulls/data/template.vtk',
                      'attachment_type': 'varifold'}}

        dataset = create_dataset(dataset_file_names, visit_ages, subject_ids, template_specifications, dimension=2, tensor_scalar_type=torch.DoubleTensor)
        assert dataset.is_time_series(), "Cannot run a geodesic regression on a non-time_series dataset."

        self.deformetrica.estimate_geodesic_regression(template_specifications, dataset,
                                                       estimator=GradientAscent,
                                                       estimator_options={'max_iterations': 100},
                                                       deformation_kernel=kernel_factory.factory(kernel_factory.Type.TORCH, kernel_width=25.0),
                                                       concentration_of_time_points=5, smoothing_kernel_width=20)

    def test_estimate_geodesic_regression_landmark_3d_surprise(self):
        dataset_file_names = [[{'skull': '../../examples/regression/landmark/3d/surprise/data/sub-F001_ses-000.vtk'},
                              {'skull': '../../examples/regression/landmark/3d/surprise/data/sub-F001_ses-005.vtk'},
                              {'skull': '../../examples/regression/landmark/3d/surprise/data/sub-F001_ses-010.vtk'},
                              {'skull': '../../examples/regression/landmark/3d/surprise/data/sub-F001_ses-015.vtk'},
                              {'skull': '../../examples/regression/landmark/3d/surprise/data/sub-F001_ses-020.vtk'},
                              {'skull': '../../examples/regression/landmark/3d/surprise/data/sub-F001_ses-025.vtk'},
                              {'skull': '../../examples/regression/landmark/3d/surprise/data/sub-F001_ses-030.vtk'},
                              {'skull': '../../examples/regression/landmark/3d/surprise/data/sub-F001_ses-035.vtk'}]]
        visit_ages = [[0, 5, 10, 15, 20, 25, 30, 35]]
        subject_ids = [['ses-000', 'ses-005', 'ses-010', 'ses-015', 'ses-020', 'ses-025', 'ses-030', 'ses-035']]
        template_specifications = {
            'skull': {'deformable_object_type': 'polyline',
                      'noise_std': 0.0035,
                      'filename': '../../examples/regression/landmark/3d/surprise/data/ForInitialization__Template__FromUser.vtk',
                      'attachment_type': 'landmark'}}

        dataset = create_dataset(dataset_file_names, visit_ages, subject_ids, template_specifications, dimension=3, tensor_scalar_type=torch.DoubleTensor)
        assert dataset.is_time_series(), "Cannot run a geodesic regression on a non-time_series dataset."

        self.deformetrica.estimate_geodesic_regression(template_specifications, dataset,
                                                       estimator=GradientAscent,
                                                       estimator_options={'max_iterations': 50, 'convergence_tolerance': 1e-5, 'initial_step_size': 1e-6},
                                                       deformation_kernel=kernel_factory.factory(kernel_factory.Type.TORCH, kernel_width=0.015),
                                                       concentration_of_time_points=1, smoothing_kernel_width=20, t0=5.5,
                                                       use_sobolev_gradient=True, dense_mode=True)

    def test_estimate_geodesic_regression_image_2d_cross(self):
        dataset_file_names = [[{'skull': '../../examples/regression/image/2d/cross/data/cross_-5.png'},
                              {'skull': '../../examples/regression/image/2d/cross/data/cross_-3.png'},
                              {'skull': '../../examples/regression/image/2d/cross/data/cross_-2.png'},
                              {'skull': '../../examples/regression/image/2d/cross/data/cross_0.png'},
                              {'skull': '../../examples/regression/image/2d/cross/data/cross_1.png'},
                              {'skull': '../../examples/regression/image/2d/cross/data/cross_3.png'}]]
        visit_ages = [[-5, -3, -2, 0, 1, 3]]
        subject_ids = [['t-5', 't-3', 't-2', 't0', 't1', 't3']]
        template_specifications = {
            'skull': {'deformable_object_type': 'image',
                      'noise_std': 0.1,
                      'filename': '../../examples/regression/image/2d/cross/data/cross_0.png',
                      'attachment_type': 'varifold'}}

        dataset = create_dataset(dataset_file_names, visit_ages, subject_ids, template_specifications, dimension=2, tensor_scalar_type=torch.DoubleTensor)
        assert dataset.is_time_series(), "Cannot run a geodesic regression on a non-time_series dataset."

        self.deformetrica.estimate_geodesic_regression(template_specifications, dataset,
                                                       estimator=GradientAscent,
                                                       estimator_options={'max_iterations': 50, 'initial_step_size': 1e-9},
                                                       deformation_kernel=kernel_factory.factory(kernel_factory.Type.TORCH, kernel_width=10.0),
                                                       concentration_of_time_points=3, freeze_template=True)

    # Registration

    def test_estimate_deterministic_registration_landmark_2d_points(self):
        dataset_file_names = [[{'pointcloud': '../../examples/registration/landmark/2d/points/data/target_points.vtk'}]]
        visit_ages = []
        subject_ids = ['target']
        template_specifications = {
            'pointcloud': {'deformable_object_type': 'landmark',
                           'noise_std': 1e-3,
                           'filename': '../../examples/registration/landmark/2d/points/data/source_points.vtk'}}

        dataset = create_dataset(dataset_file_names, visit_ages, subject_ids, template_specifications, dimension=2, tensor_scalar_type=torch.DoubleTensor)

        self.deformetrica.estimate_deterministic_atlas(template_specifications, dataset,
                                                       estimator=GradientAscent,
                                                       estimator_options={'initial_step_size': 1e-8, 'max_iterations': 100, 'max_line_search_iterations': 200},
                                                       deformation_kernel=kernel_factory.factory(kernel_factory.Type.TORCH, kernel_width=3.0),
                                                       number_of_time_points=10, freeze_template=True, freeze_control_points=True)

    def test_estimate_deterministic_registration_landmark_2d_starfish(self):
        dataset_file_names = [[{'starfish': '../../examples/registration/landmark/2d/starfish/data/starfish_target.vtk'}]]
        visit_ages = []
        subject_ids = ['target']
        template_specifications = {
            'starfish': {'deformable_object_type': 'polyline',
                         'kernel': kernel_factory.factory(kernel_factory.Type.TORCH, kernel_width=50.0),
                         'noise_std': 0.1,
                         'attachment_type': 'current',
                         'filename': '../../examples/registration/landmark/2d/starfish/data/starfish_reference.vtk'}}

        dataset = create_dataset(dataset_file_names, visit_ages, subject_ids, template_specifications, dimension=2, tensor_scalar_type=torch.DoubleTensor)

        self.deformetrica.estimate_deterministic_atlas(template_specifications, dataset,
                                                       estimator=ScipyOptimize,
                                                       estimator_options={'max_iterations': 200},
                                                       deformation_kernel=kernel_factory.factory(kernel_factory.Type.TORCH, kernel_width=30.0),
                                                       number_of_time_points=10, freeze_template=True, freeze_control_points=True)

    def test_estimate_deterministic_registration_image_2d_tetris(self):
        dataset_file_names = [[{'image': '../../examples/registration/image/2d/tetris/data/image2.png'}]]
        visit_ages = []
        subject_ids = ['target']
        template_specifications = {
            'image': {'deformable_object_type': 'image',
                      'kernel': kernel_factory.factory(kernel_factory.Type.TORCH, kernel_width=10.0),
                      'noise_std': 0.1,
                      'filename': '../../examples/registration/image/2d/tetris/data/image1.png'}}

        dataset = create_dataset(dataset_file_names, visit_ages, subject_ids, template_specifications, dimension=2, tensor_scalar_type=torch.DoubleTensor)

        self.deformetrica.estimate_deterministic_atlas(template_specifications, dataset,
                                                       estimator=GradientAscent,
                                                       deformation_kernel=kernel_factory.factory(kernel_factory.Type.TORCH, kernel_width=20.0))

    # Parallel Transport

    def test_compute_parallel_transport_image_2d_snowman(self):
        BASE_DIR='../../examples/parallel_transport/image/2d/snowman/'
        dataset_file_names = [[{'image': BASE_DIR + 'data/I1.png'}]]
        visit_ages = []
        subject_ids = ['image']
        template_specifications = {
            'image': {'deformable_object_type': 'image',
                      'noise_std': 0.05,
                      'filename':  BASE_DIR + 'data/I1.png'}}

        dataset = create_dataset(dataset_file_names, visit_ages, subject_ids, template_specifications, dimension=2, tensor_scalar_type=torch.DoubleTensor)

        self.deformetrica.compute_parallel_transport(template_specifications, dataset,
                                                     initial_control_points=BASE_DIR + 'data/Reference_progression_ControlPoints.txt',
                                                     initial_momenta=BASE_DIR + 'data/Reference_progression_Momenta.txt',
                                                     initial_momenta_to_transport=BASE_DIR + 'data/Registration_Momenta.txt',
                                                     initial_control_points_to_transport=BASE_DIR + 'data/Registration_ControlPoints.txt',
                                                     tmin=0, tmax=1,
                                                     concentration_of_time_points=10,
                                                     deformation_kernel=kernel_factory.factory(kernel_factory.Type.TORCH, kernel_width=15.0))

    # Shooting

    def test_compute_shooting_image_2d_snowman(self):
        BASE_DIR='../../examples/shooting/image/2d/snowman/'
        # dataset_file_names = [[{'image': BASE_DIR + 'data/I1.png'}]]
        dataset_file_names = []
        visit_ages = []
        subject_ids = []
        template_specifications = {
            'image': {'deformable_object_type': 'image',
                      'noise_std': 0.05,
                      'filename':  BASE_DIR + 'data/I1.png'}}

        dataset = create_dataset(dataset_file_names, visit_ages, subject_ids, template_specifications, dimension=2, tensor_scalar_type=torch.cuda.FloatTensor)

        self.deformetrica.compute_shooting(template_specifications, dataset,
                                           deformation_kernel=kernel_factory.factory(kernel_factory.Type.TORCH, kernel_width=35.0,
                                                                                     dimension=dataset.dimension,
                                                                                     tensor_scalar_type=dataset.tensor_scalar_type),
                                           initial_control_points=BASE_DIR + 'data/control_points.txt',
                                           initial_momenta=BASE_DIR + 'data/momenta.txt')


