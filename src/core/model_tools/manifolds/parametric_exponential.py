import os.path
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../../')

import torch
from pydeformetrica.src.core.model_tools.manifolds.exponential_interface import ExponentialInterface

"""
Class with a parametric inverse metric: $$g_{\theta}(q) = \sum_{i=1}^n \alpha_i \exp{-\frac {\|x-q\|^2} {2 \sigma^2}$$ 
"""

class ParametricExponential(ExponentialInterface):

    def __init__(self):
        ExponentialInterface.__init__(self)

        self.number_of_interpolation_points = None
        self.width = None
        self.interpolation_points_torch = None
        self.interpolation_values_torch = None

        self.has_closed_form = False
        self.has_closed_form_dp = True

    def inverse_metric(self, q):
        squared_distances = (self.interpolation_points_torch - q)**2.
        return torch.dot(self.interpolation_values_torch, torch.exp(-1.*squared_distances/self.width**2))

    def dp(self, q, p):
        squared_distances = (self.interpolation_points_torch - q)**2.
        A = torch.exp(-1.*squared_distances/self.width**2.)
        differences = self.interpolation_points_torch - q
        return 1./self.width**2. * torch.sum(self.interpolation_values_torch*differences*A) * p**2

    def set_parameters(self, extra_parameters):
        """
        In this case, the parameters are the interpolation values
        """
        assert extra_parameters.size() == self.interpolation_values_torch.size(),\
            "Wrong format of parameters"
        assert extra_parameters.size() == self.interpolation_points_torch.size(), \
            "Number of metric values differ from number of interpolation poins !"
        self.interpolation_values_torch = extra_parameters
        self.is_modified = True

    def project_metric_parameters(self, metric_parameters):
        for i in range(len(metric_parameters)):
            if metric_parameters[i] < 0:
                metric_parameters[i] = 0

        metric_parameters /= np.sum(metric_parameters)

        return metric_parameters


    def project_metric_parameters_gradient(self, metric_parameters_gradient):
        """
        Projection to ensure identifiability of the geodesic parametrizations.
        """
        #We project the gradient of the metric parameters onto the orthogonal of the constraint.
        orthogonal_gradient = np.ones(len(metric_parameters_gradient))
        orthogonal_gradient /= np.linalg.norm(orthogonal_gradient)
        projected_gradient = metric_parameters_gradient - np.dot(metric_parameters_gradient,
                                                orthogonal_gradient) * orthogonal_gradient
        sp = abs(np.dot(projected_gradient['metric_parameters'], orthogonal_gradient))
        assert sp < 1e-6, "Gradient incorrectly projected %f" % sp

        return projected_gradient