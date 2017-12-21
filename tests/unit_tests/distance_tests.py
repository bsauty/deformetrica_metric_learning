import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')
import numpy as np
from pydeformetrica.src.in_out.deformable_object_reader import DeformableObjectReader
from pydeformetrica.src.support.utilities.general_settings import Settings
from pydeformetrica.src.core.model_tools.attachments.multi_object_attachment import MultiObjectAttachment
from pydeformetrica.src.support.kernels.kernel_functions import create_kernel
import unittest
from torch.autograd import Variable
import torch

#Tests are done both in 2 and 3d.

class DistanceTests(unittest.TestCase):
    """
    Methods with names starting by "test" will be run
    """
    def setUp(self):
        pass

    def _read_surface_mesh(self, path):
        reader = DeformableObjectReader()
        object = reader.CreateObject(path, "SurfaceMesh")
        object.update()
        return object

    def test_varifold_distance_to_self_is_zero(self):
        surf = self._read_surface_mesh(os.path.join(Settings().unit_tests_data_dir, "hippocampus.vtk"))
        surf.update()
        points = Variable(torch.from_numpy(surf.get_data()).type(Settings().tensor_scalar_type))
        multi_attach = MultiObjectAttachment()
        kernel_width = 10.
        varifold_distance = multi_attach._varifold_distance(points, surf, surf, kernel_width).data.numpy()[0]
        self.assertTrue(abs(varifold_distance)<1e-10)

    # def test_current_distance_to_self_is_zero(self):
    #     surf = self._read_surface_mesh(os.path.join(Settings().unit_tests_data_dir, "hippocampus.vtk"))
    #     surf.update()
    #     points = Variable(torch.from_numpy(surf.get_data()).type(Settings().tensor_scalar_type))
    #     multi_attach = MultiObjectAttachment()
    #     kernel = create_kernel('exact', 10.)
    #     current_distance = multi_attach._current_distance(points, surf, surf, kernel).data.numpy()[0]
    #     self.assertTrue(abs(current_distance)<1e-10)

    def test_varifold_distance_compared_to_old_deformetrica(self):
        source = self._read_surface_mesh(os.path.join(Settings().unit_tests_data_dir, "hippocampus.vtk"))
        source.update()
        target = self._read_surface_mesh(os.path.join(Settings().unit_tests_data_dir, "hippocampus_2.vtk"))
        source.update()
        points_source = Variable(torch.from_numpy(source.get_data()).type(Settings().tensor_scalar_type))
        multi_attach = MultiObjectAttachment()
        kernel_width = 10.
        varifold_distance = multi_attach._varifold_distance(points_source, source, target, kernel_width).data.numpy()[0]
        old_deformetrica_varifold_distance = 10662.59732
        print(varifold_distance)
        self.assertTrue(abs(varifold_distance - old_deformetrica_varifold_distance)<1e-5)

    def test_current_distance_compared_to_old_deformetrica(self):
        source = self._read_surface_mesh(os.path.join(Settings().unit_tests_data_dir, "hippocampus.vtk"))
        source.update()
        target = self._read_surface_mesh(os.path.join(Settings().unit_tests_data_dir, "hippocampus_2.vtk"))
        source.update()
        points_source = Variable(torch.from_numpy(source.get_data()).type(Settings().tensor_scalar_type))
        multi_attach = MultiObjectAttachment()
        kernel_width = create_kernel('exact', 10.)
        current_distance = multi_attach._current_distance(points_source, source, target, kernel_width).data.numpy()[0]
        old_deformetrica_varifold_distance = 3657.504384
        print(current_distance)
        self.assertTrue(abs(current_distance - old_deformetrica_varifold_distance) < 1e-5)


