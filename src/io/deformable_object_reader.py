import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../')

from pydeformetrica.src.core.observations.surface_mesh import SurfaceMesh

import vtk

class DeformableObjectReader:

    """
    Creates PyDeformetrica objects from specified filename and object type.

    """

    def CreateObject(self, objectFilename, objectType):
        if objectType == 'OrientedSurfaceMesh' or objectType == 'NonOrientedSurfaceMesh':
            polyDataReader = vtk.vtkPolyDataReader()
            polyDataReader.SetFileName(objectFilename)

            object = SurfaceMesh()
            object.SetPolyData(polyDataReader.GetOutput())

        return object
