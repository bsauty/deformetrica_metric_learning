import os.path
import sys
import warnings

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../../')

from pydeformetrica.src.core.model_tools.manifolds.manifold_calculator import ManifoldCalculator
import torch


"""
An implementation of this interface must implement the inverse metric method, and optionnaly, a closed form or a closed form for dp.
Any exponential object is best used through a generic_geodesic.
"""

class ExponentialInterface():

    def __init__(self):
        self.manifold_calculator = ManifoldCalculator()

        self.number_of_time_points = 10
        self.position_t = []

        self.initial_momenta = None
        self.initial_position = None
        self.initial_velocity = None

        self.is_modified = True

        self.norm_squared = None

        self.has_closed_form = None
        self.has_closed_form_dp = None

    def get_initial_position(self):
        return self.initial_position

    def set_initial_position(self, q):
        self.initial_position = q
        self.is_modified = True

    def velocity_to_momenta(self, v):
        """
        Must be called at the initial position.
        """
        return torch.matmul(torch.inverse(self.inverse_metric(self.initial_position).view(1, 1)), v)

    def momenta_to_velocity(self, p):
        """
        Must be called at the initial position.
        """
        return torch.matmul(self.inverse_metric(self.initial_position), p)

    def set_initial_momenta(self, p):
        self.initial_momenta = p
        self.initial_velocity = self.momenta_to_velocity(p)
        self.is_modified = True

    def set_initial_velocity(self, v):
        self.initial_velocity = v
        self.initial_momenta = self.velocity_to_momenta(v)
        self.is_modified = True

    def inverse_metric(self, q):
        raise ValueError("Inverse metric must be implemented in the child classes of the exponential interface.")

    def dp(self, q, p):
        raise ValueError("Dp must be implemented in the child classes of the exponential interface. "
                         "Alternatively, the flag has_closed_form_dp must be set to off.")

    def _flow(self):
        """
        Generic flow of an exponential.
        """
        if self.initial_position is None:
            msg = "In exponential update, I am not flowing because I don't have an initial position"
            warnings.warn(msg)
        if self.has_closed_form:
            raise ValueError("Flow should not be called on a closed form exponential. Set has_closed_form to True.")
        elif self.initial_momenta is None:
            msg = "In exponential update, I am not flowing because I don't have an initial momenta"
            warnings.warn(msg)
        else:
            """
            Standard flow using the Hamiltonian equation
            if dp is not provided, autodiff is used (expensive)
            """
            if self.has_closed_form_dp:
                self.position_t = self.manifold_calculator.exponential(
                    self.initial_position, self.initial_momenta,
                    nb_steps=self.number_of_time_points,
                    inverse_metric=self.inverse_metric,
                    dp=self.dp)
            else:
                self.position_t = self.manifold_calculator.exponential(
                    self.initial_position, self.initial_momenta,
                    nb_steps=self.number_of_time_points,
                    inverse_metric=self.inverse_metric)

    def update(self):
        """
        Update the exponential object. Only way to properly flow.
        """
        if self.has_closed_form:
            raise ValueError("Update should not be called on a closed form exponential.")
        assert self.number_of_time_points > 0
        if self.is_modified:
            self._flow()
            self._update_norm_squared()
            self.is_modified = False

    def _update_norm_squared(self):
        self.norm_squared = self.manifold_calculator.hamiltonian(
            self.initial_position, self.initial_momenta, self.inverse_metric)

    def set_parameters(self, extra_parameters):
        """
        Used to set any extra parameters of the exponential object.
        """
        msg = 'Set parameters called, but not implemented ! Is this right ?'
        warnings.warn(msg)

