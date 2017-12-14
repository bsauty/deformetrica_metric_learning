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

        self.t0 = None
        self.concentration_of_time_points = 10
        self.target_times = None

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

    def get_norm(self):
        return torch.dot(self.initial_momenta.view(-1), self.kernel.convolve(
            self.initial_control_points, self.initial_control_points, self.initial_momenta).view(-1))

    def update(self):
        """
        Compute the time bounds, accordingly sets the number of points and momenta of the attribute exponentials,
        then shoot and flow them.
        """

        self._shoot()
        self._flow()

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


    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    def _shoot(self):
        """
        Computes the flow of momenta and control points
        """
        # TODO : not shoot if small momenta norm
        assert len(self.initial_control_points) > 0, "Control points not initialized in shooting"
        assert len(self.initial_momenta) > 0, "Momenta not initialized in shooting"
        # if torch.norm(self.InitialMomenta)<1e-20:
        #     self.PositionsT = [self.InitialControlPoints for i in range(self.NumberOfTimePoints)]
        #     self.InitialMomenta = [self.InitialControlPoints for i in range(self.NumberOfTimePoints)]
        self.positions_t = []
        self.momenta_t = []
        self.positions_t.append(self.initial_control_points)
        self.momenta_t.append(self.initial_momenta)
        dt = 1.0 / (self.number_of_time_points - 1.)
        # REPLACE with an hamiltonian (e.g. une classe hamiltonien)
        for i in range(self.number_of_time_points):
            dPos = self.kernel.convolve(self.positions_t[i], self.positions_t[i], self.momenta_t[i])
            dMom = self.kernel.convolve_gradient(self.momenta_t[i], self.positions_t[i])
            self.positions_t.append(self.positions_t[i] + dt * dPos)
            self.momenta_t.append(self.momenta_t[i] - dt * dMom)

            # TODO : check if it's possible to reduce overhead and keep that in CPU when pykp kernel is used.

    def _flow(self):
        """
        Flow The trajectory of the landmark points
        """
        # TODO : no flow if small momenta norm
        assert len(self.positions_t) > 0, "Shoot before flow"
        assert len(self.momenta_t) > 0, "Control points given but no momenta"
        assert len(self.landmark_points) > 0, "Please give landmark points to flow"
        # if torch.norm(self.InitialMomenta)<1e-20:
        #     self.LandmarkPointsT = [self.LandmarkPoints for i in range(self.InitialMomenta)]
        dt = 1.0 / (self.number_of_time_points - 1.)
        self.landmark_points_t = []
        self.landmark_points_t.append(self.landmark_points)
        for i in range(self.number_of_time_points):
            dPos = self.kernel.convolve(self.landmark_points_t[i], self.positions_t[i], self.momenta_t[i])
            self.landmark_points_t.append(self.landmark_points_t[i] + dt * dPos)