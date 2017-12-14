import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../')
import numpy as np
from pydeformetrica.src.support.utilities.general_settings import Settings
from vtk import vtkPolyDataWriter, vtkPoints

class Landmark:

    """
    Landmarks (i.e. labelled point sets).
    The Landmark class represents a set of labelled points. This class assumes that the source and the target
    have the same number of points with a point-to-point correspondence.

    """

    # Constructor.
    def __init__(self):
        self.poly_data = None
        self.point_coordinates = None
        self.is_modified = True
        self.bounding_box = None
        self.norm = None

    def get_number_of_points(self):
        return len(self.point_coordinates)

    # Sets the PolyData attribute, and initializes the PointCoordinates one according to the ambient space dimension.
    def set_poly_data(self, polyData):
        self.poly_data = polyData

        number_of_points = polyData.GetNumberOfPoints()
        dimension = Settings().dimension
        point_coordinates = np.zeros((number_of_points, dimension))

        for k in range(number_of_points):
            p = polyData.GetPoint(k)
            point_coordinates[k,:] = p[0:dimension]
        self.point_coordinates = point_coordinates

        self.is_modified = True

    def set_points(self, points):
        """
        Sets the list of points of the poly data, to save at the end.
        """
        self.point_coordinates = points
        vtk_points = vtkPoints()
        if (Settings().dimension == 3):
            for i in range(len(points)):
                vtk_points.InsertNextPoint((points[i,0],points[i,1],points[i,2]))
        else:
            for i in range(len(points)):
                vtk_points.InsertNextPoint((points[i,0],points[i,1],0))
        self.poly_data.SetPoints(vtk_points)

    # Gets the geometrical data that defines the landmark object, as a matrix list.
    def get_data(self):
        return self.point_coordinates

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
            self.bounding_box[d, 0] = np.min(self.point_coordinates[:, d])
            self.bounding_box[d, 1] = np.max(self.point_coordinates[:, d])

    def write(self, name):
        writer = vtkPolyDataWriter()
        writer.SetInputData(self.poly_data)
        name = os.path.join(Settings().output_dir, name)
        writer.SetFileName(name)
        writer.Update()
