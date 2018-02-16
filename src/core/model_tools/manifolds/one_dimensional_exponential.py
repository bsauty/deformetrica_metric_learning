import os.path
import sys
import numpy as np
import warnings


sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../../')

from pydeformetrica.src.core.model_tools.manifolds.manifold_calculator import ManifoldCalculator

import torch
from torch.autograd import Variable
from torch.nn import Softmax

class OneDimensionalExponential:

    def __init__(self):
        self.number_of_interpolation_points = None
        self.width = None
        self.interpolation_points_torch = None
        self.interpolation_values_torch = None
        self.manifold_calculator = ManifoldCalculator()

        self.number_of_time_points = 10
        self.position_t = []

        self.initial_momenta = None
        self.initial_position = None

        self.is_modified = True

        self.norm_squared = None

    def set_interpolation_values(self, interpolation_values):
        """
        Torch tensors
        """
        self.interpolation_values_torch = interpolation_values
        self.is_modified = True

    def get_initial_position(self):
        return self.initial_position

    def set_initial_position(self, q):
        self.initial_position = q
        self.is_modified = True

    def set_initial_momenta(self, p):
        self.initial_momenta = p
        self.is_modified = True

    def inverse_metric(self, q):
        squared_distances = (self.interpolation_points_torch - q)**2.
        return torch.dot(self.interpolation_values_torch, torch.exp(-1.*squared_distances/self.width**2))

    def dp(self, q, p):
        squared_distances = (self.interpolation_points_torch - q)**2.
        A = torch.exp(-1.*squared_distances/self.width**2.)
        differences = self.interpolation_points_torch - q
        return 1./self.width**2. * torch.sum(self.interpolation_values_torch*differences*A) * p**2

    def flow(self):
        if self.initial_position is None:
            msg = "In exponential update, I am not flowing because I don't have an initial position"
            warnings.warn(msg)
        elif self.initial_momenta is None:
            msg = "In exponential update, I am not flowing because I don't have an initial momenta"
            warnings.warn(msg)
        else:
            self.position_t = self.manifold_calculator.exponential(
                self.initial_position, self.initial_momenta,
                nb_steps=self.number_of_time_points,
                inverse_metric=self.inverse_metric,
                dp=self.dp)

    def update(self):
        """
        Update the state of the object, depending on what's needed.
        This is the only clean way to call flow on the deformation.
        """
        assert self.number_of_time_points > 0
        if self.is_modified:
            self.flow()
            self.update_norm_squared()
            self.is_modified = False

    def update_norm_squared(self):
        # Should be a torch variable always (torch.dot returns variable ?)
        self.norm_squared = self.manifold_calculator.hamiltonian(
            self.initial_position, self.initial_momenta, self.inverse_metric)

    def set_parameters(self, extra_parameters):
        """
        In this case, the parameters are the interpolation values
        """
        assert extra_parameters.size() == self.interpolation_values_torch.size(),\
            "Wrong format of parameters"
        self.interpolation_values_torch = extra_parameters
        self.is_modified = True
