import warnings

# Image readers
import PIL.Image as pimg
import nibabel as nib
import numpy as np

from core.observations.deformable_objects.image import Image
from core.observations.deformable_objects.landmarks.landmark import Landmark
from core.observations.deformable_objects.landmarks.point_cloud import PointCloud
from core.observations.deformable_objects.landmarks.poly_line import PolyLine
from core.observations.deformable_objects.landmarks.surface_mesh import SurfaceMesh
from in_out.image_functions import normalize_image_intensities
from support.utilities.general_settings import Settings

import logging
logger = logging.getLogger(__name__)
logging.getLogger('PIL').setLevel(logging.WARNING)

class DeformableObjectReader:
    """
    Creates PyDeformetrica objects from specified filename and object type.

    """

    connectivity_degrees = {'LINES': 2, 'POLYGONS': 3}

    # Create a PyDeformetrica object from specified filename and object type.
    def create_object(self, object_filename, object_type):

        if object_type.lower() in ['SurfaceMesh'.lower(), 'PolyLine'.lower(),
                                   'PointCloud'.lower(), 'Landmark'.lower()]:

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
                out_object = PointCloud()
                points = DeformableObjectReader.read_vtk_file(object_filename, extract_connectivity=False)
                out_object.set_points(points)

            elif object_type.lower() == 'Landmark'.lower():
                out_object = Landmark()
                points = DeformableObjectReader.read_vtk_file(object_filename, extract_connectivity=False)
                out_object.set_points(points)

            out_object.update()

        elif object_type.lower() == 'Image'.lower():
            if object_filename.find(".png") > 0:
                img_data = np.array(pimg.open(object_filename))
                img_affine = np.eye(Settings().dimension + 1)
                if len(img_data.shape) > 2:
                    msg = 'Multi-channel images are not managed (yet). Defaulting to the first channel.'
                    warnings.warn(msg)
                    img_data = img_data[:, :, 0]

            elif object_filename.find(".npy") > 0:
                img_data = np.load(object_filename)
                img_affine = np.eye(Settings().dimension + 1)

            elif object_filename.find(".nii") > 0:
                img = nib.load(object_filename)
                img_data = img.get_data()
                img_affine = img.affine
                assert len(img_data.shape) == 3, "Multi-channel images not available (yet!)."

            else:
                raise ValueError('Unknown image extension for file: %s' % object_filename)

            # Rescaling between 0. and 1.
            img_data, img_data_dtype = normalize_image_intensities(img_data)
            out_object = Image()
            out_object.set_intensities(img_data)
            out_object.set_affine(img_affine)
            out_object.intensities_dtype = img_data_dtype

            dimension_image = len(img_data.shape)
            if dimension_image != Settings().dimension:
                logger.warning('I am reading a {}d image but the dimension is set to {}'
                               .format(dimension_image, Settings().dimension))

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
                assert int(line[2])/(dim + 1) == nb_vertices or \
                       (int(line[2])/(dim) == nb_vertices and line[0] == 'LINES'), \
                    'Should not happen, maybe invalid vtk file or invalid dimension in model.xml ?'
                break
            else:
                #print(filename, line)
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
