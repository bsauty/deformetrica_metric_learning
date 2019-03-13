import numpy as np
import logging

from core import default
from core.observations.deformable_objects.landmarks.landmark import Landmark
from support import utilities


logger = logging.getLogger(__name__)


class PolyLine(Landmark):
    """
    Lines in 2D or 3D space
    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, dimension):
        Landmark.__init__(self, dimension)
        self.type = 'PolyLine'

        self.connectivity = None
        # All these attributes are torch tensors.
        self.centers = None
        self.normals = None

    # Clone.
    def clone(self):
        clone = PolyLine(self.dimension,)
        clone.points = np.copy(self.points)
        clone.is_modified = self.is_modified
        clone.bounding_box = self.bounding_box
        clone.norm = self.norm
        clone.connectivity = np.copy(self.connectivity)
        return clone

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def update(self):
        # self.get_centers_and_normals()
        Landmark.update(self)

    def get_centers_and_normals(self, points=None,
                                tensor_integer_type=default.tensor_integer_type,
                                tensor_scalar_type=default.tensor_scalar_type,
                                device='cpu'):
        """
        Given a new set of points, use the corresponding connectivity available in the polydata
        to compute the new normals (which are tangents in this case) and centers
        It's also a lazy initialization of those attributes !
        """
        connectivity_torch = utilities.move_data(self.connectivity, dtype=tensor_integer_type, device=device)
        if points is None:
            if self.is_modified or self.centers is None:
                torch_points_coordinates = utilities.move_data(self.points, dtype=tensor_scalar_type, device=device)
                a = torch_points_coordinates[connectivity_torch[:, 0]]
                b = torch_points_coordinates[connectivity_torch[:, 1]]
                centers = (a+b)/2.
                self.centers = centers
                self.normals = b - a
            else:
                self.centers = utilities.move_data(self.centers, device=device)
                self.normals = utilities.move_data(self.normals, device=device)
        else:
            a = points[connectivity_torch[:, 0]]
            b = points[connectivity_torch[:, 1]]
            centers = (a+b)/2.
            self.centers = centers
            self.normals = b - a
        return self.centers, self.normals
