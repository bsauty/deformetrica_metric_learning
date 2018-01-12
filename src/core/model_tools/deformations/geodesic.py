import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../../')

import torch
from torch.autograd import Variable
import numpy as np
import warnings

from pydeformetrica.src.in_out.utils import *
from pydeformetrica.src.support.utilities.general_settings import Settings
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

        self.concentration_of_time_points = None

        self.t0 = None
        self.tmax = None
        self.tmin = None

        self.control_points_t0 = None
        self.momenta_t0 = None
        self.template_data_t0 = None

        self.backward_exponential = Exponential()
        self.forward_exponential = Exponential()

        # Flags to save extra computations that have already been made in the update methods.
        self.shoot_is_modified = True
        self.flow_is_modified = True

    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################

    def set_use_rk2(self, use_rk2):
        self.backward_exponential.set_use_rk2(use_rk2)
        self.forward_exponential.set_use_rk2(use_rk2)

    def set_kernel(self, kernel):
        self.backward_exponential.kernel = kernel
        self.forward_exponential.kernel = kernel

    def set_t0(self, t0):
        self.t0 = t0
        self.shoot_is_modified = True

    def set_tmin(self, tmin):
        self.tmin = tmin
        self.shoot_is_modified = True

    def set_tmax(self, tmax):
        self.tmax = tmax
        self.shoot_is_modified = True

    def set_template_data_t0(self, td):
        self.template_data_t0 = td
        self.flow_is_modified = True

    def set_control_points_t0(self, cp):
        self.control_points_t0 = cp
        self.shoot_is_modified = True

    def set_momenta_t0(self, mom):
        self.momenta_t0 = mom
        self.shoot_is_modified = True

    def get_template_data(self, time, with_index=False):
        """
        Returns the position of the landmark points, at the given time.
        """
        assert time >= self.tmin and time <= self.tmax
        if self.shoot_is_modified or self.flow_is_modified:
            msg = "Asking for deformed template data but the geodesic was modified and not updated"
            warnings.warn(msg)

        # Backward part ------------------------------------------------------------------------------------------------
        if time <= self.t0:
            if self.backward_exponential.number_of_time_points > 1:
                step_size = (self.t0 - self.tmin) / float(self.backward_exponential.number_of_time_points - 1)
                time_index = int((time - self.tmin) / step_size + 0.5)
                if with_index:
                    return self.backward_exponential.get_template_data(time_index), \
                           self.backward_exponential.number_of_time_points - 1 - time_index
                else:
                    return self.backward_exponential.get_template_data(time_index)
            else:
                if with_index:
                    return self.backward_exponential.initial_template_data, \
                           self.backward_exponential.number_of_time_points - 1
                else:
                    return self.backward_exponential.initial_template_data

        # Forward part -------------------------------------------------------------------------------------------------
        else:
            if self.forward_exponential.number_of_time_points > 1:
                step_size = (self.tmax - self.t0) / float(self.forward_exponential.number_of_time_points - 1)
                time_index = int((time - self.t0) / step_size + 0.5)
                if with_index:
                    return self.forward_exponential.get_template_data(time_index), \
                           self.backward_exponential.number_of_time_points - 1 + time_index
                else:
                    return self.forward_exponential.get_template_data(time_index)
            else:
                if with_index:
                    return self.forward_exponential.initial_template_data, \
                           self.backward_exponential.number_of_time_points - 1
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

        assert self.t0 >= self.tmin, "tmin should be smaller than t0"
        assert self.t0 <= self.tmax, "tmax should be larger than t0"

        # Backward exponential -----------------------------------------------------------------------------------------
        delta_t = self.t0 - self.tmin
        self.backward_exponential.number_of_time_points = max(1, int(delta_t * self.concentration_of_time_points + 1.5))
        if self.shoot_is_modified:
            self.backward_exponential.set_initial_momenta(- self.momenta_t0 * delta_t)
            self.backward_exponential.set_initial_control_points(self.control_points_t0)
        if self.flow_is_modified:
            self.backward_exponential.set_initial_template_data(self.template_data_t0)
        if self.backward_exponential.number_of_time_points > 1:
            self.backward_exponential.update()

        # Forward exponential ------------------------------------------------------------------------------------------
        delta_t = self.tmax - self.t0
        self.forward_exponential.number_of_time_points = max(1, int(delta_t * self.concentration_of_time_points + 1.5))
        if self.shoot_is_modified:
            self.forward_exponential.set_initial_momenta(self.momenta_t0 * delta_t)
            self.forward_exponential.set_initial_control_points(self.control_points_t0)
        if self.flow_is_modified:
            self.forward_exponential.set_initial_template_data(self.template_data_t0)
        if self.forward_exponential.number_of_time_points > 1:
            self.forward_exponential.update()

        self.shoot_is_modified = False
        self.flow_is_modified = False

    def get_norm_squared(self):
        """
        Get the norm of the geodesic.
        """
        return self.forward_exponential.get_norm_squared()

    # Write functions --------------------------------------------------------------------------------------------------
    def write_flow(self, root_name, objects_name, objects_extension, template):

        # Initialization -----------------------------------------------------------------------------------------------
        template_data = template.get_points()

        # Backward part ------------------------------------------------------------------------------------------------
        if self.backward_exponential.number_of_time_points > 1:
            dt = (self.t0 - self.tmin) / float(self.backward_exponential.number_of_time_points - 1)

            for j, data in enumerate(self.backward_exponential.template_data_t):
                time = self.t0 - dt * j

                names = []
                for k, (object_name, object_extension) in enumerate(zip(objects_name, objects_extension)):
                    name = root_name + '__' + object_name \
                           + '__tp_' + str(self.backward_exponential.number_of_time_points - 1 - j) \
                           + ('__age_%.2f' % time) + object_extension
                    names.append(name)

                template.set_data(data.data.numpy())
                template.write(names)

        else:
            names = []
            for k, (object_name, object_extension) in enumerate(zip(objects_name, objects_extension)):
                name = root_name + '__' + object_name \
                       + '__tp_' + str(self.backward_exponential.number_of_time_points - 1) \
                       + ('__age_%.2f' % self.t0) + object_extension
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

    def parallel_transport(self, momenta_to_transport_t0, with_tangential_component=True):
        """
        :param momenta_to_transport_t0: the vector to parallel transport, given at t0 and carried at control_points_t0
        :returns: the full trajectory of the parallel transport, from tmin to tmax
        """

        if self.shoot_is_modified:
            msg = "Trying to parallel transport but the geodesic object was modified, please update before."
            warnings.warn(msg)

        if self.backward_exponential.number_of_time_points > 1:
            backward_transport = self.backward_exponential.parallel_transport(momenta_to_transport_t0,
                                                                              with_tangential_component)
        else:
            backward_transport = []

        if self.forward_exponential.number_of_time_points > 1:
            forward_transport = self.forward_exponential.parallel_transport(momenta_to_transport_t0,
                                                                            with_tangential_component)
        else:
            forward_transport = []

        return backward_transport[::-1] + forward_transport[1:]

    # def write_control_points_and_momenta_flow(self, name):
    #     """
    #     Write the flow of cp and momenta
    #     names are expected without extension
    #     """
    #     assert len(self.positions_t) == len(self.momenta_t), "Something is wrong, not as many cp as momenta in diffeo"
    #     for i in range(len(self.positions_t)):
    #         write_2D_array(self.positions_t[i].data.numpy(), name + "_Momenta_" + str(i) + ".txt")
    #         write_2D_array(self.momenta_t[i].data.numpy(), name + "_Controlpoints_" + str(i) + ".txt")

    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    def _get_times(self):
        times_backward = []
        if self.backward_exponential.number_of_time_points > 1:
            times_backward = np.linspace(
                self.t0, self.tmin, num=self.backward_exponential.number_of_time_points).tolist()

        times_forward = []
        if self.forward_exponential.number_of_time_points > 1:
            times_forward = np.linspace(
                self.t0, self.tmax, num=self.forward_exponential.number_of_time_points).tolist()

        return times_backward[::-1] + times_forward[1:]

    def _get_control_points_trajectory(self):
        if self.shoot_is_modified:
            msg = "Trying to get cp trajectory in a non updated geodesic."
            warnings.warn(msg)

        backward_control_points_t = []
        if self.backward_exponential.number_of_time_points > 1:
            backward_control_points_t = self.backward_exponential.control_points_t

        forward_control_points_t = []
        if self.forward_exponential.number_of_time_points > 1:
            forward_control_points_t = self.forward_exponential.control_points_t

        return backward_control_points_t[::-1] + forward_control_points_t[1:]

    def _get_momenta_trajectory(self):
        if self.shoot_is_modified:
            msg = "Trying to get mom trajectory in non updated geodesic."
            warnings.warn(msg)

        backward_momenta_t = []
        if self.backward_exponential.number_of_time_points > 1:
            backward_length = self.t0 - self.tmin
            backward_momenta_t = self.backward_exponential.momenta_t
            backward_momenta_t = [elt / backward_length for elt in backward_momenta_t]

        forward_momenta_t = []
        if self.forward_exponential.number_of_time_points > 1:
            forward_length = self.tmax - self.t0
            forward_momenta_t = self.forward_exponential.momenta_t
            forward_momenta_t = [elt / forward_length for elt in forward_momenta_t]

        return backward_momenta_t[::-1] + forward_momenta_t[1:]

    def _get_template_trajectory(self):
        if self.shoot_is_modified or self.flow_is_modified:
            msg = "Trying to get mom trajectory in non updated geodesic."
            warnings.warn(msg)

        backward_template_t = []
        if self.backward_exponential.number_of_time_points > 1:
            backward_template_t = self.backward_exponential.template_data_t

        forward_template_t = []
        if self.forward_exponential.number_of_time_points > 1:
            forward_template_t = self.forward_exponential.template_data_t

        return backward_template_t[::-1] + forward_template_t[1:]
