import os
import unittest

import torch
from pydeformetrica.src.core.model_tools.attachments.multi_object_attachment import MultiObjectAttachment
from pydeformetrica.src.in_out.deformable_object_reader import DeformableObjectReader
from pydeformetrica.src.support.kernels.kernel_functions import create_kernel
from pydeformetrica.src.support.utilities.general_settings import Settings
from torch.autograd import Variable


#Tests are done both in 2 and 3d.

class DistanceTests(unittest.TestCase):
    """
    Methods with names starting by "test" will be run
    """
    def setUp(self):
        pass

    def _read_surface_mesh(self, path):
        reader = DeformableObjectReader()
        object = reader.create_object(path, "SurfaceMesh")
        object.update()
        return object

    def _read_poly_line(self, path):
        reader = DeformableObjectReader()
        object = reader.create_object(path, "PolyLine")
        object.update()
        return object

    def test_surface_mesh_varifold_distance_to_self_is_zero(self):
        surf = self._read_surface_mesh(os.path.join(Settings().unit_tests_data_dir, "hippocampus.vtk"))
        points = Variable(torch.from_numpy(surf.get_points()).type(Settings().tensor_scalar_type))
        multi_attach = MultiObjectAttachment()
        kernel_width = 10.
        varifold_distance = multi_attach._varifold_distance(points, surf, surf, kernel_width).data.numpy()[0]
        self.assertTrue(abs(varifold_distance)<1e-10)

    def test_surface_mesh_current_distance_to_self_is_zero(self):
        surf = self._read_surface_mesh(os.path.join(Settings().unit_tests_data_dir, "hippocampus.vtk"))
        points = Variable(torch.from_numpy(surf.get_points()).type(Settings().tensor_scalar_type))
        multi_attach = MultiObjectAttachment()
        kernel = create_kernel('torch', 10.)
        current_distance = multi_attach._current_distance(points, surf, surf, kernel).data.numpy()[0]
        self.assertTrue(abs(current_distance)<1e-10)

    def test_varifold_distance_on_surface_mesh_is_equal_to_old_deformetrica(self):
        source = self._read_surface_mesh(os.path.join(Settings().unit_tests_data_dir, "hippocampus.vtk"))
        target = self._read_surface_mesh(os.path.join(Settings().unit_tests_data_dir, "hippocampus_2.vtk"))
        points_source = Variable(torch.from_numpy(source.get_points()).type(Settings().tensor_scalar_type))
        multi_attach = MultiObjectAttachment()
        kernel_width = 10.
        varifold_distance = multi_attach._varifold_distance(points_source, source, target, kernel_width).data.numpy()[0]
        old_deformetrica_varifold_distance = 10662.59732
        self.assertTrue(abs(varifold_distance - old_deformetrica_varifold_distance)<1e-5)

    def test_current_distance_on_surface_mesh_is_equal_to_old_deformetrica(self):
        source = self._read_surface_mesh(os.path.join(Settings().unit_tests_data_dir, "hippocampus.vtk"))
        source.update()
        target = self._read_surface_mesh(os.path.join(Settings().unit_tests_data_dir, "hippocampus_2.vtk"))
        source.update()
        points_source = Variable(torch.from_numpy(source.get_points()).type(Settings().tensor_scalar_type))
        multi_attach = MultiObjectAttachment()
        kernel_width = create_kernel('torch', 10.)
        current_distance = multi_attach._current_distance(points_source, source, target, kernel_width).data.numpy()[0]
        old_deformetrica_varifold_distance = 3657.504384
        self.assertTrue(abs(current_distance - old_deformetrica_varifold_distance) < 1e-5)

    def test_poly_line_current_distance_to_self_is_zero(self):
        Settings().dimension = 2
        self._test_poly_line_current_distance_to_self_is_zero()
        Settings().dimension = 3
        self._test_poly_line_current_distance_to_self_is_zero()

    def _test_poly_line_current_distance_to_self_is_zero(self):
        poly = self._read_poly_line(os.path.join(Settings().unit_tests_data_dir, "skull.vtk"))
        points = Variable(torch.from_numpy(poly.get_points()).type(Settings().tensor_scalar_type))
        multi_attach = MultiObjectAttachment()
        kernel = create_kernel('torch', 10.)
        current_distance = multi_attach._current_distance(points, poly, poly, kernel).data.numpy()[0]
        self.assertTrue(abs(current_distance)<1e-10)
