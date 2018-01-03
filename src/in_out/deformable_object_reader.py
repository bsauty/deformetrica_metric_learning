import os.path
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../')

from pydeformetrica.src.core.observations.deformable_objects.landmarks.surface_mesh import SurfaceMesh
from pydeformetrica.src.core.observations.deformable_objects.landmarks.poly_line import PolyLine
from pydeformetrica.src.core.observations.deformable_objects.landmarks.point_cloud import PointCloud
from pydeformetrica.src.core.observations.deformable_objects.landmarks.landmark import Landmark
from pydeformetrica.src.support.utilities.general_settings import Settings


from vtk import vtkPolyDataReader


class DeformableObjectReader:

    """
    Creates PyDeformetrica objects from specified filename and object type.

    """

    # Create a PyDeformetrica object from specified filename and object type.
    def CreateObject(self, object_filename, object_type):

        if object_type.lower() in ['SurfaceMesh'.lower(), 'PolyLine'.lower(), 'PointCloud'.lower(), 'Landmark'.lower()]:

            poly_data_reader = vtkPolyDataReader()
            poly_data_reader.SetFileName(object_filename)
            poly_data_reader.Update()

            poly_data = poly_data_reader.GetOutput()

            if object_type.lower() == 'SurfaceMesh'.lower():
                out_object = SurfaceMesh()
                out_object.set_connectivity(self._extract_connectivity(poly_data, 3))

            elif object_type.lower() == 'PolyLine'.lower():
                out_object = PolyLine()
                out_object.set_connectivity(self._extract_connectivity(poly_data, 2))

            elif object_type.lower() == 'PointCloud'.lower():
                out_object = PointCloud()

            elif object_type.lower() == 'Landmark'.lower():
                out_object = Landmark()

            out_object.set_points(self._extract_points(poly_data))
            out_object.update()

        else:
            raise RuntimeError('Unknown object type: '+object_type)

        return out_object

    @staticmethod
    def _extract_points(poly_data):
        number_of_points = poly_data.GetNumberOfPoints()
        dimension = Settings().dimension
        points = np.zeros((number_of_points, dimension))
        for k in range(number_of_points):
            p = poly_data.GetPoint(k)
            points[k, :] = p[0:dimension]
        return points

    @staticmethod
    def _extract_connectivity(poly_data, nb_points_per_face):
        """
        extract the list of faces from a poly data, and returns it as a numpy array
        nb_points_per_face is the number of points on each face
        """
        connectivity = np.zeros((poly_data.GetNumberOfCells(), nb_points_per_face))
        for i in range(poly_data.GetNumberOfCells()):
            for j in range(nb_points_per_face):
                connectivity[i, j] = poly_data.GetCell(i).GetPointId(j)
        return connectivity
