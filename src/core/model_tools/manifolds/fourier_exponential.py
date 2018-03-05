import os.path
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../../')

import torch
from pydeformetrica.src.core.model_tools.manifolds.exponential_interface import ExponentialInterface


"""
Class with a parametric inverse metric in Fourier form, with not so natural condition to ensure positivity...
"""

class FourierExponential(ExponentialInterface):

    def __init__(self):
        ExponentialInterface.__init__(self)

        self.coefficients = None
        self.fourier_coefficients = None

    def set_fourier_coefficients(self, fourier_coefficients):
        """
        Torch tensors
        """
        self.fourier_coefficients = fourier_coefficients
        self.is_modified = True

    def inverse_metric(self, q):
        kx = torch.arange(self.number_of_fourier_coefficients)
        return np.sum(self.fourier_coefficients * np.sin(kx * np.pi))

    def dp(self, q, p):
        squared_distances = (self.interpolation_points_torch - q)**2.
        A = torch.exp(-1.*squared_distances/self.width**2.)
        differences = self.interpolation_points_torch - q
        return 1./self.width**2. * torch.sum(self.interpolation_values_torch*differences*A) * p**2

    def set_parameters(self, extra_parameters):
        """
        In this case, the parameters are the fourier coefficients
        """
        print("Is this implementation right ? In Fourier exponential ?")
        assert extra_parameters.size() == self.fourier_coefficients.size(),\
            "Wrong format of parameters"
        self.fourier_coefficients = extra_parameters
        self.is_modified = True


# def f(x, coefs):
#    ...:     kx = np.array([k * x for k in range(len(coefs))])
#    ...:     return np.sum(coefs * np.sin(kx*np.pi))

# def generate_random_coefs(num=10):
#    ...:     coefs = [1.]
#    ...:     for i in range(num-1):
#    ...:         r = np.random.uniform(0., coefs[-1]/2)
#    ...:         coefs.append(r)
#    ...:     return coefs