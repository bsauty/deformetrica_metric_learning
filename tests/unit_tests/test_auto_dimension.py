import unittest

from core import default
from in_out.deformable_object_reader import DeformableObjectReader


class AutomaticDimensionDetectionTests(unittest.TestCase):
    """
    Methods with names starting by "test" will be run
    """
    def setUp(self):
        self.object_reader = DeformableObjectReader()

    def tearDown(self):
        super().tearDown()

    def test_auto_dimension_2D_vtk(self):
        _, dimension = self.object_reader.read_vtk_file('../../examples/atlas/landmark/2d/skulls/data/skull_australopithecus.vtk', extract_connectivity=False)
        self.assertEqual(2, dimension)
        _, dimension, _ = self.object_reader.read_vtk_file('../../examples/atlas/landmark/2d/skulls/data/skull_australopithecus.vtk', extract_connectivity=True)
        self.assertEqual(2, dimension)

    def test_auto_dimension_3D_vtk(self):
        _, dimension = self.object_reader.read_vtk_file('../../examples/atlas/landmark/3d/brain_structures/data/amygdala1.vtk', extract_connectivity=False)
        self.assertEqual(3, dimension)
        _, dimension, _ = self.object_reader.read_vtk_file('../../examples/atlas/landmark/3d/brain_structures/data/amygdala1.vtk', extract_connectivity=True)
        self.assertEqual(3, dimension)

    def test_auto_dimension_create_object(self):
        o = self.object_reader.create_object('../../examples/atlas/landmark/2d/skulls/data/skull_australopithecus.vtk', 'landmark', default.tensor_scalar_type)
        self.assertEqual(2, o.dimension)
        o = self.object_reader.create_object('../../examples/atlas/landmark/3d/brain_structures/data/amygdala1.vtk', 'landmark', default.tensor_scalar_type)
        self.assertEqual(3, o.dimension)
        o = self.object_reader.create_object('../../examples/atlas/image/2d/digits/data/digit_2_sample_1.png', 'image', default.tensor_scalar_type)
        self.assertEqual(2, o.dimension)
        o = self.object_reader.create_object('../../examples/registration/image/3d/brains/data/colin27.nii', 'image', default.tensor_scalar_type)
        self.assertEqual(3, o.dimension)

        o = self.object_reader.create_object('data/polyline_different_format.vtk', 'polyline', default.tensor_scalar_type)
        self.assertEqual(3, o.dimension)
