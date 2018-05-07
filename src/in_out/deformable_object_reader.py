import os.path
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../')

from pydeformetrica.src.core.observations.deformable_objects.landmarks.surface_mesh import SurfaceMesh
from pydeformetrica.src.core.observations.deformable_objects.landmarks.poly_line import PolyLine
from pydeformetrica.src.core.observations.deformable_objects.landmarks.point_cloud import PointCloud
from pydeformetrica.src.core.observations.deformable_objects.landmarks.landmark import Landmark
from pydeformetrica.src.core.observations.manifold_observations.image import Image
from pydeformetrica.src.support.utilities.general_settings import Settings

# Image readers
from PIL import Image as pil_image
import nibabel as nib

class DeformableObjectReader:
    """
    Creates PyDeformetrica objects from specified filename and object type.

    """

    connectivity_degrees = {'LINES' : 2, 'POLYGONS' : 3}

    # Create a PyDeformetrica object from specified filename and object type.
    def create_object(self, object_filename, object_type):

        if object_type.lower() in ['SurfaceMesh'.lower(), 'PolyLine'.lower(),
                                   'PointCloud'.lower(), 'Landmark'.lower()]:

            # poly_data_reader = vtkPolyDataReader()
            # poly_data_reader.SetFileName(object_filename)
            # poly_data_reader.Update()
            #
            # poly_data = poly_data_reader.GetOutput()

            if object_type.lower() == 'SurfaceMesh'.lower():
                out_object = SurfaceMesh()
                points, connectivity = DeformableObjectReader.read_vtk_file(object_filename, extract_connectivity=True)
                out_object.set_points(points)
                out_object.set_connectivity(connectivity)

            elif object_type.lower() == 'PolyLine'.lower():
                out_object = PolyLine()
                points, connectivity = DeformableObjectReader.read_vtk_file(object_filename, extract_connectivity=True)
                out_object.set_points(points)
                out_object.set_connectivity(connectivity)

            elif object_type.lower() == 'PointCloud'.lower():
                points = DeformableObjectReader.read_vtk_file(object_filename, extract_connectivity=False)
                out_object.set_points(points)

            elif object_type.lower() == 'Landmark'.lower():
                out_object = Landmark()
                points = DeformableObjectReader.read_vtk_file(object_filename, extract_connectivity=False)
                out_object.set_points(self._extract_points(poly_data))

            out_object.update()

        elif object_type.lower() == 'Image'.lower():
            if object_filename.find(".png") > 0:
                img_data = np.array(pil_image.open(object_filename), dtype=float)
                assert len(img_data.shape) == 2, "Multi-channel images not available (yet!)."

            elif object_filename.find(".npy") > 0:
                img_data = np.load(object_filename)
                if object_filename.find('mri') > 0:
                    img_data = img_data/255. # dirty hack for now

            elif object_filename.find(".nii") > 0:
                img_data = nib.load(object_filename).get_data()
                assert len(img_data.shape) == 3, "Multi-channel images not available (yet!)."

            # Rescaling between 0. and 1.
            # TODO : connect this to the xml parameter
            # img_data = (img_data-np.min(img_data))/(np.max(img_data) - np.min(img_data))
            out_object = Image()
            out_object.set_points(img_data)
            out_object.update()

        else:
            raise RuntimeError('Unknown object type: ' + object_type)

        return out_object

    @staticmethod
    def read_vtk_file(filename, extract_connectivity=False):
        """
        Routine to read  vtk files
        Probably needs new case management
        """

        dim = Settings().dimension
        with open(filename, 'r') as f:
            content = f.readlines()
        fifth_line = content[4].strip().split(' ')

        assert fifth_line[0] == 'POINTS'
        assert fifth_line[2] == 'float'

        nb_points = int(fifth_line[1])
        points = []
        line_start_connectivity = None

        # Reading the points:
        for i in range(5, len(content)):
            line = content[i].strip().split(' ')
            # Saving the position of the start for the connectivity
            if line == ['']:
                continue
            elif line[0] in ['LINES', 'POLYGONS']:
                line_start_connectivity = i
                connectivity_type = line[0]
                nb_vertices = int(line[1])
                assert int(line[2])/(dim + 1) == nb_vertices, 'Should not happen, maybe invalid vtk file ?'
                break
            else:
                # print(filename, line)
                points_for_line = np.array(line, dtype=float)
                points_for_line = points_for_line.reshape(int(len(points_for_line)/3), 3)[:, :dim]
                for p in points_for_line:
                    points.append(p)
        points = np.array(points)
        assert len(points) == nb_points, 'Something went wrong during the vtk reading'

        # Reading the connectivity, if needed.
        if extract_connectivity:
            assert line_start_connectivity is not None, 'Could not read the connectivity' \
                                                        'for the given vtk file'
            connectivity = []
            for i in range(line_start_connectivity + 1, line_start_connectivity + 1 + nb_vertices):
                line = content[i].strip().split(' ')

                assert int(line[0]) == DeformableObjectReader.connectivity_degrees[connectivity_type], \
                    'Wrong connectivity degree detected'

                connec = [int(elt) for elt in line[1:]]

                connectivity.append(connec)

            connectivity = np.array(connectivity)
            assert len(connectivity) == nb_vertices

            return points, connectivity

        else:
            return points

    # @staticmethod
    # def _extract_points(poly_data):
    #     number_of_points = poly_data.GetNumberOfPoints()
    #     dimension = Settings().dimension
    #     points = np.zeros((number_of_points, dimension))
    #     for k in range(number_of_points):
    #         p = poly_data.GetPoint(k)
    #         points[k, :] = p[0:dimension]
    #     return points
    #
    # @staticmethod
    # def _extract_connectivity(poly_data, nb_points_per_face):
    #     """
    #     extract the list of faces from a poly data, and returns it as a numpy array
    #     nb_points_per_face is the number of points on each face
    #     """
    #     connectivity = np.zeros((poly_data.GetNumberOfCells(), nb_points_per_face))
    #     for i in range(poly_data.GetNumberOfCells()):
    #         for j in range(nb_points_per_face):
    #             connectivity[i, j] = poly_data.GetCell(i).GetPointId(j)
    #     return connectivity
