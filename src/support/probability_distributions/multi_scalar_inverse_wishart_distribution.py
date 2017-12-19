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
        self.degrees_of_freedom = None
        self.scale_scalars = None

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def sample(self):
        raise RuntimeError(
            'The "sample" method is not implemented yet for the multi scalar inverse Wishart distribution.')
        pass

    def compute_log_likelihood(self, observations):
        """
        Careful: the input is a 1D array containing scalar variances.
        """
        out = 0.0
        for k in range(observations.shape[0]):
            out -= 0.5 * self.degrees_of_freedom[k] * (
                self.scale_scalars[k] / observations[k] + math.log(observations[k]))
        return out
