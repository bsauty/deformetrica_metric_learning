import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

import torch
from torch.autograd import Variable
import numpy as np
from math import sqrt

from pydeformetrica.src.support.utilities.general_settings import Settings


class MultiScalarNormalDistribution:
    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self):
        self.mean = np.zeros((1,))
        self.variance_sqrt = 1
        self.variance_inverse = 1

    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################

    def set_variance_sqrt(self, std):
        self.variance_sqrt = std
        self.variance_inverse = 1.0 / std ** 2

    def set_variance(self, var):
        self.variance_sqrt = sqrt(var)
        self.variance_inverse = 1.0 / var

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def sample(self):
        return self.mean + self.variance_sqrt * np.random.standard_normal(self.mean.shape)

    def compute_log_likelihood(self, observation):
        """
        Fully numpy method.
        Returns only the part that includes the observation argument.
        """
        assert self.mean.shape == observation.ravel().shape
        delta = observation.ravel() - self.mean
        return - 0.5 * self.variance_inverse * np.sum(delta ** 2)

    def compute_log_likelihood_torch(self, observation):
        """
        Fully torch method.
        Returns only the part that includes the observation argument.
        """
        mean = Variable(torch.from_numpy(self.mean).type(Settings().tensor_scalar_type), requires_grad=False)
        assert mean.view(-1, 1).size() == observation.view(-1, 1).size()
        delta = observation.view(-1, 1) - mean.view(-1, 1)
        return - 0.5 * self.variance_inverse * torch.sum(delta ** 2)
