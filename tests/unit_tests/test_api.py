import os
import unittest

import support.kernels as kernel_factory
from api.deformetrica import Deformetrica
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
        visit_ages = [[], [], [], [], []]
        subject_ids = ['australopithecus', 'erectus', 'habilis', 'neandertalis', 'sapiens']
        template_specifications = {
            'skull': {'deformable_object_type': 'polyline',
                      'kernel': kernel_factory.factory(kernel_factory.Type.TORCH, kernel_width=20.0),
                      'noise_std': 1.0,
                      'filename': '../../examples/atlas/landmark/2d/skulls/data/template.vtk',
                      'attachment_type': 'varifold'}}

        dataset = create_dataset(dataset_file_names, visit_ages, subject_ids, template_specifications, dimension=2)

        self.deformetrica.estimate_deterministic_atlas(template_specifications, dataset,
                                                       estimator=GradientAscent(initial_step_size=1., max_iterations=10, max_line_search_iterations=10),
                                                       deformation_kernel=kernel_factory.factory(kernel_factory.Type.TORCH, kernel_width=40.))

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
        visit_ages = [[], [], [], []]
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
                                                       estimator=ScipyOptimize(max_iterations=10),
                                                       deformation_kernel=kernel_factory.factory(kernel_factory.Type.TORCH, kernel_width=7.0),
                                                       freeze_template=False, freeze_control_points=True)

