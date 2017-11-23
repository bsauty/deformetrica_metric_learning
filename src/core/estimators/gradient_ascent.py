import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

import numpy

from pydeformetrica.src.core.estimators.abstract_estimator import *

class GradientAscent(AbstractEstimator):

    """
    GradientAscent object class.
    An estimator is an algorithm which updates the fixed effects of a statistical model.

    """

    def __init__(self):
        self.InitialStepSize = 0.001