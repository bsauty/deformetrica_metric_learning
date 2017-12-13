import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../')

from pydeformetrica.src.core.observations.deformable_objects.landmarks.surface_mesh import SurfaceMesh

from vtk import vtkPolyDataReader


class DeformableObjectReader:

    """
    Creates PyDeformetrica objects from specified filename and object type.

    """

    # Create a PyDeformetrica object from specified filename and object type.
    def CreateObject(self, objectFilename, objectType):

        if objectType.lower() == 'OrientedSurfaceMesh'.lower() \
                or objectType.lower() == 'NonOrientedSurfaceMesh'.lower():
            polyDataReader = vtkPolyDataReader()
            polyDataReader.SetFileName(objectFilename)
            polyDataReader.Update()

            obj = SurfaceMesh()
            polyData = polyDataReader.GetOutput()
            obj.SetPolyData(polyDataReader.GetOutput())
            obj.update()


        else:
            raise RuntimeError('Unknown object type: '+objectType)

        return obj
