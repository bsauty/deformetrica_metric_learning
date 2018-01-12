import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

import numpy as np
import math


class MultiScalarInverseWishartDistribution:
    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self):
        self.degrees_of_freedom = []
        self.scale_scalars = []

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def sample(self):
        raise RuntimeError(
            'The "sample" method is not implemented yet for the multi scalar inverse Wishart distribution.')
        pass

    def compute_log_likelihood(self, observations):
        """
        The input is a 1D array containing scalar variances, or a simple scalar.
        """
        assert len(self.degrees_of_freedom) == len(self.scale_scalars)

        if not isinstance(observations, np.ndarray):
            return -0.5 * self.degrees_of_freedom[0] * (self.scale_scalars[0] / observations + math.log(observations))

        else:
            assert len(self.scale_scalars) == observations.shape[0]
            out = 0.0
            for k in range(observations.shape[0]):
                out -= 0.5 * self.degrees_of_freedom[k] * (
                    self.scale_scalars[k] / observations[k] + math.log(observations[k]))
            return out
