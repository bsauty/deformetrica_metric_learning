import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

import numpy as np

from pydeformetrica.src.core.estimators.abstract_estimator import AbstractEstimator

class ScipyOptimize(AbstractEstimator):

    """
    ScipyOptimize object class.
    An estimator is an algorithm which updates the fixed effects of a statistical model.

    """

    ################################################################################
    ### Constructor:
    ################################################################################

    # def __init__(self):
    #     self.InitialStepSize = None
    #     self.MaxLineSearchIterations = None
    #
    #     self.LineSearchShrink = None
    #     self.LineSearchExpand = None
    #     self.ConvergenceTolerance = None
    #
    #     self.LogLikelihoodHistory = []


    ################################################################################
    ### Public methods:
    ################################################################################

    # Runs the scipy optimize routine and updates the statistical model.
    def Update(self):
