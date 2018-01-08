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

        self.modulation_matrix = None
        self.projected_modulation_matrix = None

    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################

    def set_use_rk2(self, use_rk2):
        self.geodesic.set_use_rk2(use_rk2)
        self.exponential.set_use_rk2(use_rk2)

    def set_control_points_t0(self, cp):
        self.geodesic.set_control_points_t0(cp)

    def set_momenta_t0(self, mom):
        self.geodesic.set_momenta_t0(mom)

    def set_template_data_t0(self, td):
        self.geodesic.set_template_data_t0(td)

    def set_kernel(self, kernel):
        self.geodesic.set_kernel(kernel)
        self.exponential.set_kernel(kernel)


    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def update(self):
        """
        Shoot and flow the geodesic, and compute the parallel transport of each column of the modulation matrix along
        this geodesic, ignoring the tangential components.
        """
        pass
