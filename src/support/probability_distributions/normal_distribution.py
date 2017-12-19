import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

import numpy as np


class NormalDistribution:
    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self):
        self.mean = np.zeros((1,))
        self.covariance = np.ones((1, 1))
        self.covariance_sqrt = np.ones((1, 1))
        self.covariance_inverse = np.ones((1, 1))
        self.covariance_log_determinant = 0

    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################

    def set_covariance(self, cov):
        self.covariance = cov
        self.covariance_sqrt = np.linalg.cholesky(cov)
        self.covariance_inverse = np.linalg.inv(cov)
        self.covariance_log_determinant = np.linalg.slogdet(cov)[1]

    def set_covariance_sqrt(self, cov_sqrt):
        self.covariance = np.dot(cov_sqrt, np.transpose(cov_sqrt))
        self.covariance_sqrt = cov_sqrt
        self.covariance_inverse = np.linalg.inv(self.covariance)
        self.covariance_log_determinant = np.linalg.slogdet(self.covariance)[1]

    def set_covariance_inverse(self, cov_inv):
        self.covariance = np.linalg.inv(cov_inv)
        self.covariance_sqrt = np.linalg.cholesky(self.covariance)
        self.covariance_inverse = cov_inv
        self.covariance_log_determinant = np.linalg.slogdet(self.covariance)[1]

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def sample(self):
        return self.mean + np.dot(self.covariance_sqrt, np.random.standard_normal(self.mean.shape))

    def compute_log_likelihood(self, observation):
        assert self.mean.shape == observation.shape
        delta = observation - self.mean
        return - 0.5 * (np.dot(delta, np.dot(self.covariance_inverse, delta)) + self.covariance_log_determinant)
