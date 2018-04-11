import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../../')

import torch
from torch.autograd import Variable
import numpy as np
import warnings
from pydeformetrica.src.in_out.array_readers_and_writers import *
from pydeformetrica.src.support.utilities.general_settings import Settings
from pydeformetrica.src.core.model_tools.deformations.exponential import Exponential
from pydeformetrica.src.core.model_tools.deformations.geodesic import Geodesic
from pydeformetrica.src.support.kernels.kernel_functions import create_kernel


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
        self.backward_extension = None
        self.forward_extension = None

        self.times = None
        self.template_points_t = None
        self.control_points_t = None

    def clone(self):
        clone = SpatiotemporalReferenceFrame()

        clone.geodesic = self.geodesic.clone()
        clone.exponential = self.exponential.clone()

        if self.modulation_matrix_t0 is not None:
            clone.modulation_matrix_t0 = self.modulation_matrix_t0.clone()
        if self.projected_modulation_matrix_t0 is not None:
            clone.projected_modulation_matrix_t0 = self.projected_modulation_matrix_t0.clone()
        if self.projected_modulation_matrix_t is not None:
            clone.projected_modulation_matrix_t = [elt.clone() for elt in self.projected_modulation_matrix_t]
        clone.number_of_sources = self.number_of_sources

        clone.transport_is_modified = self.transport_is_modified
        clone.backward_extension = self.backward_extension
        clone.forward_extension = self.forward_extension

        clone.times = self.times
        if self.template_points_t is not None:
            clone.template_points_t = {key: [elt.clone() for elt in value]
                                       for key, value in self.template_points_t.item()}
        if self.control_points_t is not None:
            clone.control_points_t = [elt.clone() for elt in self.control_points_t]


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

    def get_concentration_of_time_points(self):
        return self.geodesic.concentration_of_time_points

    def set_concentration_of_time_points(self, ctp):
        self.geodesic.concentration_of_time_points = ctp

    def set_number_of_time_points(self, ntp):
        self.exponential.number_of_time_points = ntp

    def set_template_points_t0(self, td):
        self.geodesic.set_template_points_t0(td)

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

    def get_tmin(self):
        return self.geodesic.get_tmin()

    def set_tmin(self, tmin, optimize=False):
        self.geodesic.set_tmin(tmin, optimize)
        self.backward_extension = self.geodesic.backward_extension

    def get_tmax(self):
        return self.geodesic.get_tmax()

    def set_tmax(self, tmax, optimize=False):
        self.geodesic.set_tmax(tmax, optimize)
        self.forward_extension = self.geodesic.forward_extension

    def get_template_points_exponential(self, time, sources):

        # Assert for coherent length of attribute lists.
        assert len(self.template_points_t[list(self.template_points_t.keys())[0]]) == len(self.control_points_t) \
               == len(self.projected_modulation_matrix_t) == len(self.times)

        # Initialize the returned exponential.
        exponential = Exponential()
        exponential.kernel = create_kernel(self.exponential.kernel.kernel_type, self.exponential.kernel.kernel_width)
        exponential.number_of_time_points = self.exponential.number_of_time_points
        exponential.use_rk2 = self.exponential.use_rk2

        # Deal with the special case of a geodesic reduced to a single point.
        if len(self.times) == 1:
            print('>> The spatiotemporal reference frame geodesic seems to be reduced to a single point.')
            exponential.set_initial_template_points({key: value[0] for key, value in self.template_points_t})
            exponential.set_initial_control_points(self.control_points_t[0])
            exponential.set_initial_momenta(torch.mm(self.projected_modulation_matrix_t[0],
                                                     sources.unsqueeze(1)).view(self.geodesic.momenta_t0.size()))
            return exponential

        # Standard case.
        index, weight_left, weight_right = self._get_interpolation_index_and_weights(time)
        template_points = {key: weight_left * value[index - 1] + weight_right * value[index]
                           for key, value in self.template_points_t}
        control_points = weight_left * self.control_points_t[index - 1] + weight_right * self.control_points_t[index]
        modulation_matrix = weight_left * self.projected_modulation_matrix_t[index - 1] \
                            + weight_right * self.projected_modulation_matrix_t[index]
        space_shift = torch.mm(modulation_matrix, sources.unsqueeze(1)).view(self.geodesic.momenta_t0.size())

        exponential.set_initial_template_points(template_points)
        exponential.set_initial_control_points(control_points)
        exponential.set_initial_momenta(space_shift)
        return exponential

    def get_template_points(self, time, sources):

        # Assert for coherent length of attribute lists.
        assert len(self.template_points_t[list(self.template_points_t.keys())[0]]) == len(self.control_points_t) \
               == len(self.projected_modulation_matrix_t) == len(self.times)

        # Deal with the special case of a geodesic reduced to a single point.
        if len(self.times) == 1:
            print('>> The spatiotemporal reference frame geodesic seems to be reduced to a single point.')
            self.exponential.set_initial_template_points({key: value[0] for key, value in self.template_points_t})
            self.exponential.set_initial_control_points(self.control_points_t[0])
            self.exponential.set_initial_momenta(torch.mm(self.projected_modulation_matrix_t[0],
                                                     sources.unsqueeze(1)).view(self.geodesic.momenta_t0.size()))
            self.exponential.update()
            return self.exponential.get_template_points()

        # Standard case.
        index, weight_left, weight_right = self._get_interpolation_index_and_weights(time)
        template_points = {key: weight_left * value[index - 1] + weight_right * value[index]
                           for key, value in self.template_points_t.items()}
        control_points = weight_left * self.control_points_t[index - 1] + weight_right * self.control_points_t[index]
        modulation_matrix = weight_left * self.projected_modulation_matrix_t[index - 1] \
                            + weight_right * self.projected_modulation_matrix_t[index]
        space_shift = torch.mm(modulation_matrix, sources.unsqueeze(1)).view(self.geodesic.momenta_t0.size())

        self.exponential.set_initial_template_points(template_points)
        self.exponential.set_initial_control_points(control_points)
        self.exponential.set_initial_momenta(space_shift)
        self.exponential.update()
        return self.exponential.get_template_points()

    def _get_interpolation_index_and_weights(self, time):
        for index in range(1, len(self.times)):
            if time.data.cpu().numpy()[0] - self.times[index] < 0: break
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
        self.template_points_t = self.geodesic._get_template_points_trajectory()
        self.control_points_t = self.geodesic._get_control_points_trajectory()

        if self.transport_is_modified:
            # Initializes the projected_modulation_matrix_t attribute size.
            self.projected_modulation_matrix_t = \
                [Variable(torch.zeros(self.modulation_matrix_t0.size()).type(Settings().tensor_scalar_type),
                          requires_grad=False) for _ in range(len(self.control_points_t))]

            # Transport each column, ignoring the tangential components.
            for s in range(self.number_of_sources):
                space_shift_t0 = self.modulation_matrix_t0[:, s].contiguous().view(self.geodesic.momenta_t0.size())
                space_shift_t = self.geodesic.parallel_transport(space_shift_t0, with_tangential_component=False)

                # Set the result correctly in the projected_modulation_matrix_t attribute.
                for t, space_shift in enumerate(space_shift_t):
                    self.projected_modulation_matrix_t[t][:, s] = space_shift.view(-1)

            self.transport_is_modified = False
            self.backward_extension = 0
            self.forward_extension = 0

        elif self.backward_extension > 0 or self.forward_extension > 0:

            # Initializes the extended projected_modulation_matrix_t variable.
            projected_modulation_matrix_t_extended = \
                [Variable(torch.zeros(self.modulation_matrix_t0.size()).type(Settings().tensor_scalar_type),
                          requires_grad=False) for _ in range(len(self.control_points_t))]

            # Transport each column, ignoring the tangential components.
            for s in range(self.number_of_sources):
                space_shift_t = [elt[:, s].contiguous().view(self.geodesic.momenta_t0.size())
                                 for elt in self.projected_modulation_matrix_t]
                # print(len(self.control_points_t))
                space_shift_t = self.geodesic.extend_parallel_transport(
                    space_shift_t, self.backward_extension, self.forward_extension, with_tangential_component=False)

                for t, space_shift in enumerate(space_shift_t):
                    projected_modulation_matrix_t_extended[t][:, s] = space_shift.view(-1)

            self.projected_modulation_matrix_t = projected_modulation_matrix_t_extended
            self.backward_extension = 0
            self.forward_extension = 0

        assert len(self.template_points_t[list(self.template_points_t.keys())[0]]) == len(self.control_points_t) \
                == len(self.times) == len(self.projected_modulation_matrix_t), \
            "That's weird: len(self.template_points_t[list(self.template_points_t.keys())[0]]) = %d, " \
            "len(self.control_points_t) = %d, len(self.times) = %d,  len(self.projected_modulation_matrix_t) = %d" % \
            (len(self.template_points_t[list(self.template_points_t.keys())[0]]), len(self.control_points_t),
             len(self.times), len(self.projected_modulation_matrix_t))

    ####################################################################################################################
    ### Writing methods:
    ####################################################################################################################

    def write(self, root_name, objects_name, objects_extension, template, template_data,
              write_adjoint_parameters=False, write_exponential_flow=False):

        # Write the geodesic -------------------------------------------------------------------------------------------
        self.geodesic.write(root_name, objects_name, objects_extension, template, write_adjoint_parameters)

        # Write the exp-parallel curves --------------------------------------------------------------------------------
        times = self.geodesic._get_times()
        for t, (time, modulation_matrix) in enumerate(zip(times, self.projected_modulation_matrix_t)):
            for s in range(self.number_of_sources):
                space_shift = modulation_matrix[:, s].contiguous().view(self.geodesic.momenta_t0.size())
                self.exponential.set_initial_template_points({key: value[t] for key, value in self.template_points_t})
                self.exponential.set_initial_control_points(self.control_points_t[t])
                self.exponential.set_initial_momenta(space_shift)
                self.exponential.update()
                deformed_points = self.exponential.get_template_points()
                deformed_data = template.get_deformed_data(deformed_points, template_data)

                names = []
                for k, (object_name, object_extension) in enumerate(zip(objects_name, objects_extension)):
                    name = root_name + '__IndependentComponent_' + str(s) + '__' + object_name + '__tp_' + str(t) \
                           + ('__age_%.2f' % time) + object_extension
                    names.append(name)
                template.write(names, {key: value.data.numpy() for key, value in deformed_data.items()})

                # Massive writing.
                if write_exponential_flow:
                    names = []
                    for k, (object_name, object_extension) in enumerate(zip(objects_name, objects_extension)):
                        name = root_name + '__IndependentComponent_' + str(s) + '__' + object_name + '__tp_' + str(t) \
                               + ('__age_%.2f' % time) + '__ExponentialFlow'
                        names.append(name)
                    self.exponential.write_flow(names, objects_extension, template, write_adjoint_parameters)

