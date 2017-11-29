import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../')

from pydeformetrica.src.core.observations.deformable_objects.landmarks.surface_mesh import SurfaceMesh

from vtk import vtkPolyDataReader


class DeformableObjectReader:

    """
    Creates PyDeformetrica objects from specified filename and object type.

    """

    def CreateObject(self, objectFilename, objectType):

        if objectType == 'OrientedSurfaceMesh' or objectType == 'NonOrientedSurfaceMesh':
            polyDataReader = vtkPolyDataReader()
            polyDataReader.SetFileName(objectFilename)
            polyDataReader.Update()

            object = SurfaceMesh()
            polyData = polyDataReader.GetOutput()
            object.SetPolyData(polyDataReader.GetOutput())

        else:
            raise RuntimeError('Unknown object type: '+objectType)

        return object
