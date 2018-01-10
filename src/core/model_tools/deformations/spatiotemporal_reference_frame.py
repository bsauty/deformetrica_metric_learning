import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../../')

import torch
import numpy as np
import warnings
from pydeformetrica.src.in_out.utils import *
from pydeformetrica.src.core.model_tools.deformations.exponential import Exponential
from pydeformetrica.src.core.model_tools.deformations.exponential import Geodesic


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
        deformed_points, time_index = self.geodesic.get_template_data(time, with_index=True)
        space_shift = torch.mm(self.projected_modulation_matrix_t[time_index], sources).view(
            self.geodesic.momenta_t0.shape)
        self.exponential.set_initial_template_data(deformed_points)
        self.exponential.set_initial_control_points(self.control_points_t[time_index])
        self.exponential.set_initial_momenta(space_shift)
        self.exponential.update()
        self.exponential.get_template_data()

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

        # Convenient attribute for later use.
        self.control_points_t = self.geodesic.backward_exponential.control_points_t[::-1] + \
                                self.geodesic.backward_exponential.control_points_t[1:]

        if self.transport_is_modified:
            # Transport each column, ignoring the tangential components.
            for s in range(self.number_of_sources):
                space_shift_t0 = self.projected_modulation_matrix_t0[:, s].view(self.geodesic.momenta_t0.shape)
                assert False  # Need careful check here.
                space_shift_t = self.geodesic.parallel_transport(space_shift_t0, with_tangential_component=False)
                self.projected_modulation_matrix_t = [elt.view(-1) for elt in space_shift_t]
            self.transport_is_modified = False