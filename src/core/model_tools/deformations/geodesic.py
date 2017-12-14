import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../../')

import torch

from pydeformetrica.src.in_out.utils import *
from pydeformetrica.src.core.model_tools.deformations.exponential import Exponential


class Geodesic:
    """
    Control-point-based LDDMM geodesic.
    See "Morphometry of anatomical shape complexes with dense deformations and sparse parameters",
    Durrleman et al. (2013).

    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self):

        self.concentration_of_time_points = 10

        self.t0 = None
        self.tmax = None
        self.tmin = None

        self.control_points_t0 = None
        self.momenta_t0 = None
        self.template_data_t0 = None

        self.backward_exponential = Exponential()
        self.forward_exponential = Exponential()


    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################

    def set_kernel(self, kernel):
        self.backward_exponential.kernel = kernel
        self.forward_exponential.kernel = kernel

    def get_landmark_points(self, time_index=None):
        """
        Returns the position of the landmark points, at the given time_index in the Trajectory
        """
        if time_index == None:
            return self.landmark_points_t[self.number_of_time_points]
        return self.landmark_points_t[time_index]

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def update(self):
        """
        Compute the time bounds, accordingly sets the number of points and momenta of the attribute exponentials,
        then shoot and flow them.
        """

        # Backward exponential -----------------------------------------------------------------------------------------
        delta_t = self.t0 - self.tmin
        self.backward_exponential.number_of_time_points = max(0, int(delta_t * self.concentration_of_time_points))
        self.backward_exponential.initial_momenta = - self.momenta_t0 * delta_t
        self.backward_exponential.initial_control_points = self.control_points_t0
        self.backward_exponential.initial_template_data = self.template_data_t0
        self.backward_exponential.update()

        # Forward exponential ------------------------------------------------------------------------------------------
        delta_t = self.tmax - self.t0
        self.backward_exponential.number_of_time_points = max(0, int(delta_t * self.concentration_of_time_points))
        self.backward_exponential.initial_momenta = self.momenta_t0 * delta_t
        self.backward_exponential.initial_control_points = self.control_points_t0
        self.backward_exponential.initial_template_data = self.template_data_t0
        self.backward_exponential.update()

    def get_norm(self):
        return torch.dot(self.momenta_t0.view(-1), self.backward_exponential.kernel.convolve(
            self.control_points_t0, self.control_points_t0, self.momenta_t0).view(-1))


    # Write functions --------------------------------------------------------------------------------------------------
    def write_flow(self, objects_names, objects_extensions, template):
        for i in range(self.number_of_time_points):
            # names = [objects_names[i]+"_t="+str(i)+objects_extensions[j] for j in range(len(objects_name))]
            names = []
            for j, elt in enumerate(objects_names):
                names.append(elt + "_t=" + str(i) + objects_extensions[j])
            deformedPoints = self.landmark_points_t[i]
            aux_points = template.get_data()
            template.set_data(deformedPoints.data.numpy())
            template.write(names)
            # restauring state of the template object for further computations
            template.set_data(aux_points)

    def write_control_points_and_momenta_flow(self, name):
        """
        Write the flow of cp and momenta
        names are expected without extension
        """
        assert len(self.positions_t) == len(self.momenta_t), "Something is wrong, not as many cp as momenta in diffeo"
        for i in range(len(self.positions_t)):
            write_2D_array(self.positions_t[i].data.numpy(), name + "_Momenta_" + str(i) + ".txt")
            write_2D_array(self.momenta_t[i].data.numpy(), name + "_Controlpoints_" + str(i) + ".txt")
