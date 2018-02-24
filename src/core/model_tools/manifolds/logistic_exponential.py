import os.path
import sys
import warnings

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../../')

from pydeformetrica.src.core.model_tools.manifolds.manifold_calculator import ManifoldCalculator

import torch

"""
Exponential on \R for 1/(q**2(1-q)) metric i.e. logistic curves.
"""

class LogisticExponential:

    def __init__(self):
        self.manifold_calculator = ManifoldCalculator()
        self.position_t = []
        self.initial_velocity = None
        self.initial_position = None
        self.initial_momenta = None
        self.is_modified = True
        self.norm_squared = None

    def get_initial_position(self):
        return self.initial_position

    def set_initial_position(self, q):
        self.initial_position = q
        self.is_modified = True

    def set_initial_momenta(self, p):
        self.initial_momenta = p
        self.initial_velocity = torch.matmul(self.inverse_metric(self.initial_position), p)
        self.is_modified = True

    def set_initial_velocity(self, v):
        self.initial_velocity = v
        self.initial_momenta = torch.matmul(1./self.inverse_metric(self.initial_position), v)
        self.is_modified = True

    def inverse_metric(self, q):
        """
        inverse metric so that geodesics are logistic curves.
        """
        return (q*(1-q))**2

    def closed_form(self, q, v, t):
        return 1./(1 + (1/q - 1) * torch.exp(-1.*v/(q * (1-q)) * t))

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
                closed_form=self.closed_form)

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

    def set_parameters(self):
        pass

