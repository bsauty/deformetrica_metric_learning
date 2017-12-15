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

    def get_template_data(self, time):
        """
        Returns the position of the landmark points, at the given time.
        """
        assert time >= self.tmin and time <= self.tmax

        # Backward part ------------------------------------------------------------------------------------------------
        if time <= self.t0:
            if self.backward_exponential.number_of_time_points > 1:
                time_index = int(self.concentration_of_time_points * (self.t0 - time)
                                 / float(self.backward_exponential.number_of_time_points - 1) + 0.5)
                return self.backward_exponential.get_template_data(time_index)
            else:
                return self.backward_exponential.initial_template_data

        # Forward part -------------------------------------------------------------------------------------------------
        else:
            if self.forward_exponential.number_of_time_points > 1:
                step_size = (self.tmax - self.t0) / float(self.forward_exponential.number_of_time_points - 1)
                time_index = int((time - self.t0) / step_size + 0.5)
                return self.forward_exponential.get_template_data(time_index)
            else:
                return self.forward_exponential.initial_template_data

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
        self.backward_exponential.number_of_time_points = max(1, delta_t * int(self.concentration_of_time_points + 1.5))
        self.backward_exponential.initial_momenta = - self.momenta_t0 * delta_t
        self.backward_exponential.initial_control_points = self.control_points_t0
        self.backward_exponential.initial_template_data = self.template_data_t0
        self.backward_exponential.update()

        # Forward exponential ------------------------------------------------------------------------------------------
        delta_t = self.tmax - self.t0
        self.forward_exponential.number_of_time_points = max(1, int(delta_t * self.concentration_of_time_points + 1.5))
        self.forward_exponential.initial_momenta = self.momenta_t0 * delta_t
        self.forward_exponential.initial_control_points = self.control_points_t0
        self.forward_exponential.initial_template_data = self.template_data_t0
        self.forward_exponential.update()

    def get_norm(self):
        return torch.dot(self.momenta_t0.view(-1), self.backward_exponential.kernel.convolve(
            self.control_points_t0, self.control_points_t0, self.momenta_t0).view(-1))

    # Write functions --------------------------------------------------------------------------------------------------
    def write_flow(self, root_name, objects_name, objects_extension, template):

        # Initialization -----------------------------------------------------------------------------------------------
        template_data = template.get_data()

        # Backward part ------------------------------------------------------------------------------------------------
        if self.backward_exponential.number_of_time_points > 1:
            dt = (self.t0 - self.tmin) / float(self.backward_exponential.number_of_time_points - 1)

            for j, data in enumerate(self.backward_exponential.template_data_t):
                time = self.t0 - dt * j

                names = []
                for k, (object_name, object_extension) in enumerate(zip(objects_name, objects_extension)):
                    name = root_name + '__' + object_name \
                           + '__tp_' + str(self.backward_exponential.number_of_time_points - 1 - j) \
                           + ('__age_%.2f' % time) + objects_extension
                    names.append(name)

                template.set_data(data.data.numpy())
                template.write(names)

        else:
            names = []
            for k, (object_name, object_extension) in enumerate(zip(objects_name, objects_extension)):
                name = root_name + '__' + object_name + '__tp_0' + ('__age_%.2f' % self.t0) + object_extension
                names.append(name)
            template.set_data(self.template_data_t0.data.numpy())
            template.write(names)

        # Forward part -------------------------------------------------------------------------------------------------
        if self.forward_exponential.number_of_time_points > 1:
            dt = (self.tmax - self.t0) / float(self.forward_exponential.number_of_time_points - 1)

            for j, data in enumerate(self.forward_exponential.template_data_t[1:], 1):
                time = self.t0 + dt * j

                names = []
                for k, (object_name, object_extension) in enumerate(zip(objects_name, objects_extension)):
                    name = root_name + '__' + object_name \
                           + '__tp_' + str(self.backward_exponential.number_of_time_points - 1 + j) \
                           + ('__age_%.2f' % time) + object_extension
                    names.append(name)

                template.set_data(data.data.numpy())
                template.write(names)

        # Finalization ------------------------------------------------------------------------------------------------
        template.set_data(template_data)

        # def write_control_points_and_momenta_flow(self, name):
        #     """
        #     Write the flow of cp and momenta
        #     names are expected without extension
        #     """
        #     assert len(self.positions_t) == len(self.momenta_t), "Something is wrong, not as many cp as momenta in diffeo"
        #     for i in range(len(self.positions_t)):
        #         write_2D_array(self.positions_t[i].data.numpy(), name + "_Momenta_" + str(i) + ".txt")
        #         write_2D_array(self.momenta_t[i].data.numpy(), name + "_Controlpoints_" + str(i) + ".txt")
