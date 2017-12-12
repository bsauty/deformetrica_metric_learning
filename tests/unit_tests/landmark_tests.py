import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')
import numpy as np
from pydeformetrica.src.in_out.deformable_object_reader import DeformableObjectReader
from pydeformetrica.src.support.utilities.general_settings import Settings
import unittest

class LandmarkTests(unittest.TestCase):
    """
    Methods with names starting by "test" will be run
    """
    def _read_surface_mesh(self, path):
        reader = DeformableObjectReader()
        object = reader.CreateObject(path, "OrientedSurfaceMesh")
        object.Update()
        return object

    def test_read_surface_mesh(self):
        surface_mesh = self._read_surface_mesh(os.path.join(Settings().UnitTestsDataDir, "hippocampus.vtk"))
        points = surface_mesh.GetData()
        point0 = np.array([148.906, 136.813, 58.2132])
        point1 = np.array([149.69, 133.984, 63.2745])
        self.assertTrue(np.allclose(point0, points[0], rtol=1e-05, atol=1e-08))
        self.assertTrue(np.allclose(point1, points[1], rtol=1e-05, atol=1e-08))


    def test_set_points_surface_mesh(self):
        """
        Reads a vtk
        Set new point coordinates using SetPoints
        Asserts the points sent by GetData of the object are the new points
        """
        surface_mesh = self._read_surface_mesh(os.path.join(Settings().UnitTestsDataDir, "hippocampus.vtk"))
        points = surface_mesh.GetData()
        random_shift = np.random.uniform(0,1,points.shape)
        deformed_points = points + random_shift
        surface_mesh.SetPoints(deformed_points)
        deformed_points_2 = surface_mesh.GetData()
        self.assertTrue(np.allclose(deformed_points, deformed_points_2, rtol=1e-05, atol=1e-08))


