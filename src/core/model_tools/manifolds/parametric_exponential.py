import os.path
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../../')

import torch
from torch.autograd import Variable
from pydeformetrica.src.core.model_tools.manifolds.exponential_interface import ExponentialInterface
from pydeformetrica.src.support.utilities.general_settings import Settings

"""
Class with a parametric inverse metric: $$g_{\theta}(q) = \sum_{i=1}^n \alpha_i \exp{-\frac {\|x-q\|^2} {2 \sigma^2}$$ 
The metric_parameters for this class is the set of symmetric positive definite matrices from which we interpolate. 
It is a (nb_points, dimension*(dimension+1)/2) tensor, enumerated by lines.
"""

class ParametricExponential(ExponentialInterface):

    def __init__(self):
        ExponentialInterface.__init__(self)

        self.number_of_interpolation_points = None
        self.width = None
        # List of points in the space, from which the metric is interpolated
        self.interpolation_points_torch = None
        # Tensor, shape (number of points, dimension, dimension)
        self.interpolation_values_torch = None

        self.diagonal_indices = None

        self.has_closed_form = False
        self.has_closed_form_dp = True

    def inverse_metric(self, q):
        squared_distances = ((self.interpolation_points_torch - q)**2.)
        return torch.sum(self.interpolation_values_torch
                         *torch.exp(-1.*squared_distances/self.width**2))

    def dp(self, q, p):
        squared_distances = (self.interpolation_points_torch - q)**2.
        A = torch.exp(-1.*squared_distances/self.width**2.)
        differences = self.interpolation_points_torch - q
        return 1./self.width**2. * torch.sum(self.interpolation_values_torch*differences*A) * p**2

    def set_parameters(self, extra_parameters):
        """
        In this case, the parameters are the interpolation values
        """

        dim = self.interpolation_points_torch.size()[1]
        size = extra_parameters.size()
        assert size[0] == self.interpolation_points_torch.size()[0]
        assert size[1] == dim * (dim + 1) /2
        symmetric_matrices = ParametricExponential.uncholeskify(extra_parameters, dim)

        self.interpolation_values_torch = symmetric_matrices
        self.is_modified = True

    @staticmethod
    def uncholeskify(l, dim):
        """
        Takes a tensor l of shape (n_cp, dimension*(dimension+1)/2)
        and returns out a tensor of shape (n_cp, dimension, dimension)
        such that out[i] = Upper(l[i]).transpose() * Upper(l[i])
        """
        print("Uncholeskify needs to be checked !")
        out = Variable(torch.from_numpy(np.zeros((l.size()[0], dim, dim))).type(Settings().tensor_scalar_type))
        for i in range(l.size()[0]):
            aux = Variable(torch.from_numpy(np.zeros((dim, dim))).type(Settings().tensor_scalar_type))
            aux[torch.triu(torch.ones(dim, dim)) == 1] = l[i]
            out[i] = aux
        return out * torch.transpose(out, 1, 2)

    # @staticmethod
    # def choleskify(l):
    #     print("Choleskify needs to be checked !")
    #     return l[:, torch.triu(torch.ones(3, 3)) == 1]

    def _get_diagonal_indices(self):
        if self.diagonal_indices is None:
            diagonal_indices = []
            spacing = 0
            pos_in_line = 0
            for j in range(self.interpolation_values_torch.size()[1] - 1, -1, -1):
                if pos_in_line == spacing:
                    spacing += 1
                    pos_in_line = 0
                    diagonal_indices.append(j)
            self.diagonal_indices = np.array(diagonal_indices)

        return self.diagonal_indices

    def project_metric_parameters(self, metric_parameters):
        """
        :param metric_parameters: numpy array of shape (number of points, dimension*(dimension+1)/2)
        we must check positivity of the diagonal coefficients of the lower matrices.
        :return:
        """
        diagonal_indices = self._get_diagonal_indices()

        # Positivity for each diagonal coefficient.
        for i in range(len(metric_parameters)):
            for j in diagonal_indices:
                if metric_parameters[i][j] < 0:
                    metric_parameters[i][j] = 0

        # Sum to one for each diagonal coefficient.
        for j in diagonal_indices:
            # Should it be necessary?
            metric_parameters[:, j] /= np.sum(metric_parameters[:, j])

        return metric_parameters

    def project_metric_parameters_gradient(self, metric_parameters_gradient):
        """
        Projection to ensure identifiability of the geodesic parametrizations.
        """
        orthogonal_gradient = np.ones(len(metric_parameters_gradient))
        orthogonal_gradient /= np.linalg.norm(orthogonal_gradient)

        diagonal_indices = self._get_diagonal_indices()

        out = metric_parameters_gradient

        for j in diagonal_indices:
            out[:, j] = out[:, j] - np.dot(out[:, j], orthogonal_gradient)

        print("Check this too !")

        return out