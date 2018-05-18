import os.path

import numpy as np
import torch
from torch.autograd import Variable

from support.utilities.general_settings import Settings


class Landmark:
    """
    Landmarks (i.e. labelled point sets).
    The Landmark class represents a set of labelled points. This class assumes that the source and the target
    have the same number of points with a point-to-point correspondence.

    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    # Constructor.
    def __init__(self):
        self.type = 'Landmark'
        self.points = None  # Numpy array.
        self.is_modified = True
        self.bounding_box = None
        self.norm = None

    # Clone.
    def clone(self):
        clone = Landmark()
        clone.points = np.copy(self.points)
        clone.is_modified = self.is_modified
        clone.bounding_box = self.bounding_box
        clone.norm = self.norm
        return clone

    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################

    def get_number_of_points(self):
        return len(self.points)

    def set_points(self, points):
        """
        Sets the list of points of the poly data, to save at the end.
        """
        self.is_modified = True
        self.points = points

    # Gets the geometrical data that defines the landmark object, as a matrix list.
    def get_points(self):
        return self.points

    def get_points_torch(self):
        return Variable(torch.from_numpy(self.points).type(Settings().tensor_scalar_type))

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    # Update the relevant information.
    def update(self):
        if self.is_modified:
            self.update_bounding_box()
            self.is_modified = False

    # Compute a tight bounding box that contains all the landmark data.
    def update_bounding_box(self):
        dimension = Settings().dimension
        self.bounding_box = np.zeros((dimension, 2))
        for d in range(dimension):
            self.bounding_box[d, 0] = np.min(self.points[:, d])
            self.bounding_box[d, 1] = np.max(self.points[:, d])

    def write(self, name, points=None):
        connec_names = {2: 'LINES', 3: 'POLYGONS'}
        if points is None:
            points = self.points

        with open(os.path.join(Settings().output_dir, name), 'w') as f:
            s = '# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET POLYDATA\nPOINTS {} float\n'.format(len(self.points))
            f.write(s)
            for p in points:
                str_p = [str(elt) for elt in p]
                if len(p) == 2:
                    str_p.append(str(0.))
                s = ' '.join(str_p) + '\n'
                f.write(s)

            if self.connectivity is not None:
                connectivity_numpy = self.connectivity.cpu().numpy()
                a, connec_degree = connectivity_numpy.shape
                s = connec_names[connec_degree] + ' {} {}\n'.format(a, a * (connec_degree+1))
                f.write(s)
                for face in connectivity_numpy:
                    s = str(connec_degree) + ' ' + ' '.join([str(elt) for elt in face]) + '\n'
                    f.write(s)