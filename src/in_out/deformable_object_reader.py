import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../')

from pydeformetrica.src.core.observations.deformable_objects.landmarks.surface_mesh import SurfaceMesh
from pydeformetrica.src.core.observations.deformable_objects.landmarks.poly_line import PolyLine
from vtk import vtkPolyDataReader


class DeformableObjectReader:

    """
    Creates PyDeformetrica objects from specified filename and object type.

    """

    # Create a PyDeformetrica object from specified filename and object type.
    def CreateObject(self, objectFilename, objectType):

        if objectType.lower() == 'SurfaceMesh'.lower():
            polyDataReader = vtkPolyDataReader()
            polyDataReader.SetFileName(objectFilename)
            polyDataReader.Update()

            object = SurfaceMesh()
            polyData = polyDataReader.GetOutput()
            object.set_poly_data(polyDataReader.GetOutput())
            object.update()

        elif objectType.lower() == 'PolyLine'.lower():
            polyDataReader = vtkPolyDataReader()
            polyDataReader.SetFileName(objectFilename)
            polyDataReader.Update()

            object = PolyLine()
            polyData = polyDataReader.GetOutput()
            object.set_poly_data(polyDataReader.GetOutput())
            object.update()


        else:
            raise RuntimeError('Unknown object type: '+objectType)

        return object
