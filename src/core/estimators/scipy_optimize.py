import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

import numpy as np
from scipy.optimize import minimize

from pydeformetrica.src.core.estimators.abstract_estimator import AbstractEstimator

class ScipyOptimize(AbstractEstimator):

    """
    ScipyOptimize object class.
    An estimator is an algorithm which updates the fixed effects of a statistical model.

    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self):
        self.FixedEffectsShape = None
    #     self.InitialStepSize = None
    #     self.MaxLineSearchIterations = None
    #
    #     self.LineSearchShrink = None
    #     self.LineSearchExpand = None
    #
    #     self.LogLikelihoodHistory = []


    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def Update(self):
        """
        Runs the scipy optimize routine and updates the statistical model.
        """

        # Initialization -----------------------------------------------------------------------------------------------
        # Dictionary of the structured parameters of the model (numpy arrays) that will be optimized.
        fixedEffects = self.StatisticalModel.GetFixedEffects()

        # Dictionary of the shapes of the model parameters.
        self.FixedEffectsShape = {key: value.shape for key, value in fixedEffects.items()}

        # Dictionary of linearized parameters.
        theta = {key: value.flatten() for key, value in fixedEffects.items()}

        # 1D numpy array that concatenates the linearized model parameters.
        x0 = np.concatenate([value for value in theta.values()])

        # Main loop ----------------------------------------------------------------------------------------------------
        self.CurrentIteration = 1
        result = minimize(self._cost_and_derivative, x0,
                          method='L-BFGS-B', jac=True, callback=self._callback,
                          options={
                              'maxiter': self.MaxIterations,
                              'ftol': self.ConvergenceTolerance,
                              'maxcor': 10,               # Number of previous gradients used to approximate the Hessian
                              'disp': True
                          })

        # Write --------------------------------------------------------------------------------------------------------
        self.Write(result.x)

    def Print(self):
        print('')
        print('------------------------------------- Iteration: ' + str(self.CurrentIteration)
              + ' -------------------------------------')

    def Write(self, x):
        """
        Save the results contained in x.
        """
        fixedEffects = self._unvectorize_fixed_effects(x)
        self.StatisticalModel.SetFixedEffects(fixedEffects)
        self.StatisticalModel.Write(self.Dataset)

    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    def _cost_and_derivative(self, x):

        # Recover the structure of the parameters ----------------------------------------------------------------------
        fixedEffects = self._unvectorize_fixed_effects(x)

        # Call the model method ----------------------------------------------------------------------------------------
        attachement, regularity, gradient = self.StatisticalModel.ComputeLogLikelihood(
            self.Dataset, fixedEffects, None, None, with_grad=True)

        # Prepare the outputs: notably linearize and concatenates the gradient -----------------------------------------
        cost = - attachement - regularity
        gradient = - np.concatenate([value.flatten() for value in gradient.values()])

        return cost, gradient

    def _unvectorize_fixed_effects(self, x):
        """
        Recover the structure of the parameters
        """
        fixedEffects = {}
        cursor = 0
        for key, shape in self.FixedEffectsShape.items():
            length = reduce(lambda x, y: x*y, shape) # Python 3: see https://stackoverflow.com/questions/13840379/how-can-i-multiply-all-items-in-a-list-together-with-python
            fixedEffects[key] = x[cursor:cursor+length].reshape(shape)
            cursor += length
        return fixedEffects

    def _callback(self, x):
        # if not (self.CurrentIteration % self.PrintEveryNIters): self.Print()
        if not (self.CurrentIteration % self.SaveEveryNIters): self.Write(x)

        self.CurrentIteration += 1














