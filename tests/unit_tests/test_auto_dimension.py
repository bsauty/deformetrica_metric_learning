import os
import unittest

from core import default
from in_out.deformable_object_reader import DeformableObjectReader
from unit_tests import unit_tests_data_dir


class AutomaticDimensionDetectionTests(unittest.TestCase):
    """
    Methods with names starting by "test" will be run
    """
    def setUp(self):
        self.object_reader = DeformableObjectReader()

    def tearDown(self):
        super().tearDown()

    def test_auto_dimension_2D_vtk(self):
        _, dimension, _ = self.object_reader.read_vtk_file(os.path.join(unit_tests_data_dir, 'bonhomme.vtk'), extract_connectivity=True)
        self.assertEqual(2, dimension)
        _, dimension = self.object_reader.read_vtk_file(os.path.join(unit_tests_data_dir, 'point_cloud.vtk'), extract_connectivity=False)
        self.assertEqual(2, dimension)
        _, dimension, _ = self.object_reader.read_vtk_file(os.path.join(unit_tests_data_dir, 'skull.vtk'), extract_connectivity=True)
        self.assertEqual(2, dimension)

    def test_auto_dimension_3D_vtk(self):
        _, dimension, _ = self.object_reader.read_vtk_file(os.path.join(unit_tests_data_dir, 'hippocampus.vtk'), extract_connectivity=True)
        self.assertEqual(3, dimension)
        _, dimension, _ = self.object_reader.read_vtk_file(os.path.join(unit_tests_data_dir, 'hippocampus_2.vtk'), extract_connectivity=True)
        self.assertEqual(3, dimension)

    def test_auto_dimension_create_object(self):
        o = self.object_reader.create_object(os.path.join(
            unit_tests_data_dir, 'bonhomme.vtk'), 'landmark',
            default.tensor_scalar_type, default.tensor_integer_type)
        self.assertEqual(2, o.dimension)
        o = self.object_reader.create_object(os.path.join(
            unit_tests_data_dir, 'point_cloud.vtk'), 'landmark',
            default.tensor_scalar_type, default.tensor_integer_type)
        self.assertEqual(2, o.dimension)
        o = self.object_reader.create_object(os.path.join(
            unit_tests_data_dir, 'skull.vtk'), 'polyline',
            default.tensor_scalar_type, default.tensor_integer_type)
        self.assertEqual(2, o.dimension)
        o = self.object_reader.create_object(os.path.join(
            unit_tests_data_dir, 'digit_2_sample_1.png'), 'image',
            default.tensor_scalar_type, default.tensor_integer_type)
        self.assertEqual(2, o.dimension)
        o = self.object_reader.create_object(os.path.join(
            unit_tests_data_dir, 'hippocampus.vtk'), 'SurfaceMesh',
            default.tensor_scalar_type, default.tensor_integer_type)
        self.assertEqual(3, o.dimension)
        o = self.object_reader.create_object(os.path.join(
            unit_tests_data_dir, 'polyline_different_format.vtk'), 'polyline',
            default.tensor_scalar_type, default.tensor_integer_type)
        self.assertEqual(3, o.dimension)
