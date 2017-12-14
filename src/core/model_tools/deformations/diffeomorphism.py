import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../../')

import torch


class Diffeomorphism:
    """
    Control-point-based LDDMM diffeomophism, that transforms the template objects according to initial control points
    and momenta parameters.
    See "Morphometry of anatomical shape complexes with dense deformations and sparse parameters",
    Durrleman et al. (2013).

    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self):
        self.kernel = None
        self.number_of_time_points = 10
        self.t0 = 0.
        self.tN = 1.
        # Initial position of control points
        self.initial_control_points = None
        # Initial momenta
        self.initial_momenta = None
        # Control points trajectory
        self.positions_t = None
        # Momenta trajectory
        self.momenta_t = None
        # Trajectory of the whole vertices of Landmark type at different time steps.
        self.landmark_points_t = None
        # Initial landmark points
        self.landmark_points = None

    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################

    def set_initial_control_points(self, initial_control_points):
        self.initial_control_points = initial_control_points

    def set_initial_momenta(self, initial_momenta):
        self.initial_momenta = initial_momenta

    def set_landmark_points(self, landmark_points):
        self.landmark_points = landmark_points

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

    def shoot(self):
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
        dt = (self.tN - self.t0) / (self.number_of_time_points - 1.)
        # REPLACE with an hamiltonian (e.g. une classe hamiltonien)
        for i in range(self.number_of_time_points):
            dPos = self.kernel.convolve(self.positions_t[i], self.positions_t[i], self.momenta_t[i])
            dMom = self.kernel.convolve_gradient(self.momenta_t[i], self.positions_t[i])
            self.positions_t.append(self.positions_t[i] + dt * dPos)
            self.momenta_t.append(self.momenta_t[i] - dt * dMom)

        #TODO : check if it's possible to reduce overhead and keep that in CPU when pykp kernel is used.

    def flow(self):
        """
        Flow The trajectory of the landmark points
        """
        # TODO : no flow if small momenta norm
        assert len(self.positions_t) > 0, "Shoot before flow"
        assert len(self.momenta_t) > 0, "Control points given but no momenta"
        assert len(self.landmark_points) > 0, "Please give landmark points to flow"
        # if torch.norm(self.InitialMomenta)<1e-20:
        #     self.LandmarkPointsT = [self.LandmarkPoints for i in range(self.InitialMomenta)]
        dt = (self.tN - self.t0) / (self.number_of_time_points - 1.)
        self.landmark_points_t = []
        self.landmark_points_t.append(self.landmark_points)
        for i in range(self.number_of_time_points):
            dPos = self.kernel.convolve(self.landmark_points_t[i], self.positions_t[i], self.momenta_t[i])
            self.landmark_points_t.append(self.landmark_points_t[i] + dt * dPos)

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
