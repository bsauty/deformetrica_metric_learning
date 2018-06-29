import os
import torch
import unittest

import support.kernels as kernel_factory
from api.deformetrica import Deformetrica
from core import default
from core.estimators.gradient_ascent import GradientAscent
from core.estimators.scipy_optimize import ScipyOptimize
from in_out.dataset_functions import create_dataset


class API(unittest.TestCase):

    def setUp(self):
        self.deformetrica = Deformetrica(output_dir=os.path.join(os.path.dirname(__file__), 'output'))

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

        dataset = create_dataset(dataset_file_names, visit_ages, subject_ids, template_specifications, dimension=2, tensor_scalar_type=torch.FloatTensor)

        self.deformetrica.estimate_deterministic_atlas(template_specifications, dataset,
                                                       estimator=GradientAscent,
                                                       estimator_options={'initial_step_size': 1., 'max_iterations': 10, 'max_line_search_iterations': 10},
                                                       deformation_kernel=kernel_factory.factory(kernel_factory.Type.TORCH, kernel_width=40.0))

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
                                                       estimator_options={'max_iterations': 10},
                                                       deformation_kernel=kernel_factory.factory(kernel_factory.Type.TORCH, kernel_width=7.0),
                                                       freeze_template=False, freeze_control_points=True)

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
