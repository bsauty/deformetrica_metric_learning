import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../')
import numpy as np
from pydeformetrica.src.support.utilities.general_settings import Settings
from vtk import vtkPolyDataWriter, vtkPoints, vtkPolyData, vtkCellArray, vtkIdList
import torch
from torch.autograd import Variable

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
        # Points attribute is a numpy array !
        self.points = None
        self.is_modified = True
        self.bounding_box = None
        self.norm = None

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

    def write(self, name):
        # We re-construct the whole poly data.
        out = vtkPolyData()
        cells = vtkCellArray()
        points = vtkPoints()

        # Building the points vtk object
        if Settings().dimension == 3:
            for i in range(len(self.points)):
                points.InsertPoint(i, self.points[i])
        else:
            for i in range(len(self.points)):
                points.InsertPoint(i, np.concatenate([self.points[i], [0.]]))

        out.SetPoints(points)


        # Building the cells vtk object
        try:
            # We try to get the connectivity attribute (to save one implementation of write in the child classes
            if self.connectivity is not None:
                for face in self.connectivity.numpy():
                    vil = vtkIdList()
                    for k in face:
                        vil.InsertNextId(int(k))
                    cells.InsertNextCell(vil)
            out.SetPolys(cells)

        except AttributeError:
            pass

        writer = vtkPolyDataWriter()
        writer.SetInputData(out)
        name = os.path.join(Settings().output_dir, name)
        writer.SetFileName(name)
        writer.Update()
