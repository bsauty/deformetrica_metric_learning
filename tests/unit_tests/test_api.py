import os
import unittest

import support.kernels as kernel_factory
from api.deformetrica import Deformetrica
from core.estimators.gradient_ascent import GradientAscent
from in_out.dataset_functions import create_dataset


class API(unittest.TestCase):

    def setUp(self):
        self.deformetrica = Deformetrica(output_dir=os.curdir)

    def test_estimate_deterministic_atlas(self):
        dataset_file_names = [[{'skull': '../../examples/atlas/landmark/2d/skulls/data/skull_australopithecus.vtk'}],
                              [{'skull': '../../examples/atlas/landmark/2d/skulls/data/skull_erectus.vtk'}],
                              [{'skull': '../../examples/atlas/landmark/2d/skulls/data/skull_habilis.vtk'}],
                              [{'skull': '../../examples/atlas/landmark/2d/skulls/data/skull_neandertalis.vtk'}],
                              [{'skull': '../../examples/atlas/landmark/2d/skulls/data/skull_sapiens.vtk'}]]
        visit_ages = [[], [], [], [], []]
        subject_ids = ['australopithecus', 'erectus', 'habilis', 'neandertalis', 'sapiens']
        template_specifications = {
            'skull': {'deformable_object_type': 'polyline', 'kernel_type': 'torch', 'kernel_width': 20.0,
                      'noise_std': 1.0,
                      'filename': '../../examples/atlas/landmark/2d/skulls/data/template.vtk',
                      'attachment_type': 'varifold'}}

        # xml_parameters = XmlParameters()
        # xml_parameters.dataset_filenames = dataset_file_names
        # xml_parameters.visit_ages = visit_ages
        # xml_parameters.subject_ids = subject_ids
        # xml_parameters.template_specifications = template_specifications
        # # xml_parameters.dimension = 2
        # xml_parameters.initial_cp_spacing = 40.
        # # xml_parameters.optimization_method_type = 'gradientascent'
        # xml_parameters.deformation_kernel_width = template_specifications['skull']['kernel_width']

        # Settings().dimension = 2

        dataset = create_dataset(dataset_file_names, visit_ages, subject_ids, template_specifications, dimension=2)

        self.deformetrica.estimate_deterministic_atlas(template_specifications, dataset,
                                                       estimator=GradientAscent(initial_step_size=1.),
                                                       deformation_kernel=kernel_factory.factory(kernel_factory.Type.TORCH, kernel_width=1.),
                                                       smoothing_kernel_width=1.)

