import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../../')
from pydeformetrica.src.in_out.utils import *
import torch


class Exponential:
    """
    Control-point-based LDDMM exponential, that transforms the template objects according to initial control points
    and momenta parameters.
    See "Morphometry of anatomical shape complexes with dense deformations and sparse parameters",
    Durrleman et al. (2013).

    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self):
        self.kernel = None
        self.number_of_time_points = None

        # Initial position of control points
        self.initial_control_points = None
        # Control points trajectory
        self.control_points_t = None

        # Initial momenta
        self.initial_momenta = None
        # Momenta trajectory
        self.momenta_t = None

        # Initial template data
        self.initial_template_data = None
        # Trajectory of the whole vertices of landmark type at different time steps.
        self.template_data_t = None

    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################

    def get_template_data(self, time_index=None):
        """
        Returns the position of the landmark points, at the given time_index in the Trajectory
        """
        if time_index is None:
            return self.template_data_t[- 1]
        return self.template_data_t[time_index]

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def update(self):
        """
        Shoot and flow.
        """
        assert self.number_of_time_points > 0
        if self.number_of_time_points > 1:
            self._shoot()
            self._flow()

    def get_norm(self):
        return torch.dot(self.initial_momenta.view(-1), self.kernel.convolve(
            self.initial_control_points, self.initial_control_points, self.initial_momenta).view(-1))

    # Write functions --------------------------------------------------------------------------------------------------
    def write_flow(self, objects_names, objects_extensions, template):
        for j, data in enumerate(self.template_data_t):
            # names = [objects_names[i]+"_t="+str(i)+objects_extensions[j] for j in range(len(objects_name))]
            names = []
            for k, elt in enumerate(objects_names): names.append(elt + "_t=" + str(j) + objects_extensions[k])
            aux_points = template.get_data()
            template.set_data(data.data.numpy())
            template.write(names)
            # restauring state of the template object for further computations
            template.set_data(aux_points)

    def write_control_points_and_momenta_flow(self, name):
        """
        Write the flow of cp and momenta
        names are expected without extension
        """
        assert len(self.control_points_t) == len(self.momenta_t), \
            "Something is wrong, not as many cp as momenta in diffeo"
        for j, control_points, momenta in enumerate(zip(self.control_points_t, self.momenta_t)):
            write_2D_array(control_points.data.numpy(), name + "__control_points_" + str(j) + ".txt")
            write_2D_array(momenta.data.numpy(), name + "__momenta_" + str(j) + ".txt")

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
        self.control_points_t = []
        self.momenta_t = []
        self.control_points_t.append(self.initial_control_points)
        self.momenta_t.append(self.initial_momenta)
        dt = 1.0 / float(self.number_of_time_points - 1)
        # REPLACE with an hamiltonian (e.g. une classe hamiltonien)
        for i in range(self.number_of_time_points - 1):
            dPos = self.kernel.convolve(self.control_points_t[i], self.control_points_t[i], self.momenta_t[i])
            dMom = self.kernel.convolve_gradient(self.momenta_t[i], self.control_points_t[i])
            self.control_points_t.append(self.control_points_t[i] + dt * dPos)
            self.momenta_t.append(self.momenta_t[i] - dt * dMom)

            # TODO : check if it's possible to reduce overhead and keep that in CPU when pykp kernel is used.

    def _flow(self):
        """
        Flow The trajectory of the landmark points
        """
        # TODO : no flow if small momenta norm
        assert len(self.control_points_t) > 0, "Shoot before flow"
        assert len(self.momenta_t) > 0, "Control points given but no momenta"
        assert len(self.initial_template_data) > 0, "Please give landmark points to flow"
        # if torch.norm(self.InitialMomenta)<1e-20:
        #     self.LandmarkPointsT = [self.LandmarkPoints for i in range(self.InitialMomenta)]
        dt = 1.0 / float(self.number_of_time_points - 1)
        self.template_data_t = []
        self.template_data_t.append(self.initial_template_data)
        for i in range(self.number_of_time_points - 1):
            dPos = self.kernel.convolve(self.template_data_t[i], self.control_points_t[i], self.momenta_t[i])
            self.template_data_t.append(self.template_data_t[i] + dt * dPos)
