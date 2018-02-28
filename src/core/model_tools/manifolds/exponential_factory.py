import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../../')

from pydeformetrica.src.core.model_tools.manifolds.parametric_exponential import ParametricExponential
from pydeformetrica.src.core.model_tools.manifolds.logistic_exponential import LogisticExponential
from pydeformetrica.src.core.model_tools.manifolds.fourier_exponential import FourierExponential


"""
Reads a dictionary of parameters, and returns the corresponding exponential object.
"""

class ExponentialFactory:
    def __init__(self):
        self.manifold_type = None
        self.manifold_parameters = None

    def set_manifold_type(self, manifold_type):
        self.manifold_type = manifold_type

    def set_parameters(self, manifold_parameters):
        self.manifold_parameters = manifold_parameters

    def create(self):
        """
        Returns an exponential for a manifold of a given type, using the parameters
        """
        if self.manifold_type == 'parametric':
            out = ParametricExponential()
            out.width = self.manifold_parameters['width']
            out.number_of_interpolation_points = self.manifold_parameters['interpolation_points_torch'].size()[0]
            out.interpolation_points_torch = self.manifold_parameters['interpolation_points_torch']
            out.interpolation_values_torch = self.manifold_parameters['interpolation_values_torch']
            return out

        if self.manifold_type == 'fourier':
            out = FourierExponential()
            out.fourier_coefficients = self.manifold_parameters['fourier_coefficients_torch']
            out.number_of_fourier_coefficients = out.fourier_coefficients.size()[0]
            return out

        if self.manifold_type == 'logistic':
            out = LogisticExponential()
            return out

        raise ValueError("Unrecognized manifold type in exponential factory")