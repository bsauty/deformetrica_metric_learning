import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')
import numpy as np
from pydeformetrica.src.in_out.deformable_object_reader import DeformableObjectReader
from pydeformetrica.src.support.utilities.general_settings import Settings
import unittest

class SurfaceMeshTests(unittest.TestCase):
    """
    Methods with names starting by "test" will be run
    """
    def setUp(self):
        #first 3 points of the hippocampus.vtk file
        self.points = [np.array([148.906, 136.813, 58.2132]), np.array([149.69, 133.984, 63.2745]), np.array([156.384, 134.243, 63.4685])]
        #first triangle of the hippocampus.vtk file
        self.first_triangle = np.array([4,44,42])

    def _read_surface_mesh(self, path):
        reader = DeformableObjectReader()
        object = reader.CreateObject(path, "SurfaceMesh")
        object.update()
        return object

    def test_read_surface_mesh(self):
        """
        Reads an example vtk file and checks a few points and triangles
        """
        surface_mesh = self._read_surface_mesh(os.path.join(Settings().unit_tests_data_dir, "hippocampus.vtk"))

        points = surface_mesh.get_points()
        self.assertTrue(np.allclose(self.points[0], points[0], rtol=1e-05, atol=1e-08))
        self.assertTrue(np.allclose(self.points[1], points[1], rtol=1e-05, atol=1e-08))
        self.assertTrue(np.allclose(self.points[2], points[2], rtol=1e-05, atol=1e-08))

        other_first_triangle = surface_mesh.connectivity[0].numpy()
        self.assertTrue(np.allclose(self.first_triangle, other_first_triangle))


    def test_set_points_surface_mesh(self):
        """
        Reads a vtk
        Set new point coordinates using SetPoints
        Asserts the points sent by GetData of the object are the new points
        """
        surface_mesh = self._read_surface_mesh(os.path.join(Settings().unit_tests_data_dir, "hippocampus.vtk"))
        points = surface_mesh.get_points()
        random_shift = np.random.uniform(0,1,points.shape)
        deformed_points = points + random_shift
        surface_mesh.set_points(deformed_points)
        deformed_points_2 = surface_mesh.get_points()
        self.assertTrue(np.allclose(deformed_points, deformed_points_2, rtol=1e-05, atol=1e-08))

    def test_centers_and_normals(self):
        """
        Tests the computation of centers and normals on the hippocampus, on all triangles
        """
        surface_mesh = self._read_surface_mesh(os.path.join(Settings().unit_tests_data_dir, "hippocampus.vtk"))
        pts = surface_mesh.get_points()
        triangles = surface_mesh.connectivity.numpy()
        centers, normals = surface_mesh.get_centers_and_normals()
        for i,triangle in enumerate(triangles):
            pts_triangle = [pts[j] for j in triangle]
            center = np.mean(pts_triangle, 0)
            normal = np.cross(pts_triangle[1]-pts_triangle[0],pts_triangle[2]-pts_triangle[0])/2.
            self.assertTrue(np.allclose(center, centers.data.numpy()[i]))
            self.assertTrue(np.allclose(normal, normals.data.numpy()[i], rtol=1e-04, atol=1e-07))
