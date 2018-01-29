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
from pydeformetrica.src.core.model_tools.deformations.geodesic import Geodesic


class SpatiotemporalReferenceFrame:
    """
    Control-point-based LDDMM spatio-temporal reference frame, based on exp-parallelization.
    See "Learning distributions of shape trajectories from longitudinal datasets: a hierarchical model on a manifold
    of diffeomorphisms", Bône et al. (2018), in review.

    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self):
        self.geodesic = Geodesic()
        self.exponential = Exponential()

        self.modulation_matrix_t0 = None
        self.projected_modulation_matrix_t0 = None
        self.projected_modulation_matrix_t = None
        self.number_of_sources = None
        self.transport_is_modified = True

        self.times = None
        self.get_template_data_t = None
        self.control_points_t = None

    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################

    def set_use_rk2(self, use_rk2):
        self.geodesic.set_use_rk2(use_rk2)
        self.exponential.set_use_rk2(use_rk2)

    def set_kernel(self, kernel):
        self.geodesic.set_kernel(kernel)
        self.exponential.set_kernel(kernel)

    def get_kernel_width(self):
        return self.exponential.kernel.kernel_width

    def set_concentration_of_time_points(self, ctp):
        self.geodesic.concentration_of_time_points = ctp

    def set_number_of_time_points(self, ntp):
        self.exponential.number_of_time_points = ntp

    def set_template_data_t0(self, td):
        self.geodesic.set_template_data_t0(td)

    def set_control_points_t0(self, cp):
        self.geodesic.set_control_points_t0(cp)
        self.transport_is_modified = True

    def set_momenta_t0(self, mom):
        self.geodesic.set_momenta_t0(mom)
        self.transport_is_modified = True

    def set_modulation_matrix_t0(self, mm):
        self.modulation_matrix_t0 = mm
        self.number_of_sources = mm.size()[1]
        self.transport_is_modified = True

    def set_t0(self, t0):
        self.geodesic.set_t0(t0)
        self.transport_is_modified = True

    def set_tmin(self, tmin):
        self.geodesic.set_tmin(tmin)
        self.transport_is_modified = True

    def set_tmax(self, tmax):
        self.geodesic.set_tmax(tmax)
        self.transport_is_modified = True

    def get_template_data(self, time, sources):
        index, weight_left, weight_right = self._get_interpolation_index_and_weights(time)
        template_data = weight_left * self.template_data_t[index - 1] + weight_right * self.template_data_t[index]
        control_points = weight_left * self.control_points_t[index - 1] + weight_right * self.control_points_t[index]
        modulation_matrix = weight_left * self.projected_modulation_matrix_t[index - 1] \
                            + weight_right * self.projected_modulation_matrix_t[index]
        space_shift = torch.mm(modulation_matrix, sources.unsqueeze(1)).view(self.geodesic.momenta_t0.size())

        self.exponential.set_initial_template_data(template_data)
        self.exponential.set_initial_control_points(control_points)
        self.exponential.set_initial_momenta(space_shift)
        self.exponential.update()
        return self.exponential.get_template_data()

    def _get_interpolation_index_and_weights(self, time):
        for index in range(1, len(self.times)):
            if time.data.numpy()[0] - self.times[index] < 0: break
        weight_left = (self.times[index] - time) / (self.times[index] - self.times[index - 1])
        weight_right = (time - self.times[index - 1]) / (self.times[index] - self.times[index - 1])
        return index, weight_left, weight_right

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def update(self):
        """
        Update the geodesic, and compute the parallel transport of each column of the modulation matrix along
        this geodesic, ignoring the tangential components.
        """
        # Update the geodesic.
        self.geodesic.update()

        # Convenient attributes for later use.
        self.times = self.geodesic._get_times()
        self.template_data_t = self.geodesic._get_template_data_trajectory()
        self.control_points_t = self.geodesic._get_control_points_trajectory()

        if self.transport_is_modified:
            # Initializes the projected_modulation_matrix_t attribute size.
            self.projected_modulation_matrix_t = \
                [Variable(torch.zeros(self.modulation_matrix_t0.size()).type(Settings().tensor_scalar_type),
                          requires_grad=False)] * len(self.control_points_t)

            # Transport each column, ignoring the tangential components.
            for s in range(self.number_of_sources):
                space_shift_t0 = self.modulation_matrix_t0[:, s].contiguous().view(self.geodesic.momenta_t0.size())
                space_shift_t = self.geodesic.parallel_transport(space_shift_t0, with_tangential_component=False)

                # Set the result correctly in the projected_modulation_matrix_t attribute.
                for t, space_shift in enumerate(space_shift_t):
                    self.projected_modulation_matrix_t[t][:, s] = space_shift.view(-1)

            self.projected_modulation_matrix_t = self.projected_modulation_matrix_t
            self.transport_is_modified = False

    ####################################################################################################################
    ### Writing methods:
    ####################################################################################################################

    def write(self, root_name, objects_name, objects_extension, template):
        # Write the geodesic -------------------------------------------------------------------------------------------
        self.geodesic.write(root_name, objects_name, objects_extension, template)

        # Write the exp-parallel curves --------------------------------------------------------------------------------
        # Initialization.
        template_data_memory = template.get_points()

        # Core loop.
        times = self.geodesic._get_times()
        for t, (time, modulation_matrix) in enumerate(zip(times, self.projected_modulation_matrix_t)):
            for s in range(self.number_of_sources):
                space_shift = modulation_matrix[:, s].contiguous().view(self.geodesic.momenta_t0.size())
                self.exponential.set_initial_template_data(self.template_data_t[t])
                self.exponential.set_initial_control_points(self.control_points_t[t])
                self.exponential.set_initial_momenta(space_shift)
                self.exponential.update()
                deformed_points = self.exponential.get_template_data()

                names = []
                for k, (object_name, object_extension) in enumerate(zip(objects_name, objects_extension)):
                    name = root_name + '__IndependentComponent_' + str(s) + '__' + object_name + '__tp_' + str(t) \
                           + ('__age_%.2f' % time) + object_extension
                    names.append(name)
                template.set_data(deformed_points.data.numpy())
                template.write(names)

        # Finalization.
        template.set_data(template_data_memory)



