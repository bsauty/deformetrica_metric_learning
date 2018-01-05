import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

import numpy as np


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
