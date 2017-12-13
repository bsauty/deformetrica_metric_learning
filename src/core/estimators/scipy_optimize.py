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
        self.fixed_effects_shape = None

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

    def update(self):
        """
        Runs the scipy optimize routine and updates the statistical model.
        """

        # Initialization -----------------------------------------------------------------------------------------------
        # Dictionary of the structured parameters of the model (numpy arrays) that will be optimized.
        fixed_effects = self.StatisticalModel.get_fixed_effects()

        # Dictionary of the shapes of the model parameters.
        self.fixed_effects_shape = {key: value.shape for key, value in fixed_effects.items()}

        # Dictionary of linearized parameters.
        theta = {key: value.flatten() for key, value in fixed_effects.items()}

        # 1D numpy array that concatenates the linearized model parameters.
        x0 = np.concatenate([value for value in theta.values()])

        # Main loop ----------------------------------------------------------------------------------------------------
        self.CurrentIteration = 1
        print('')

        result = minimize(self._cost_and_derivative, x0,
                          method='L-BFGS-B', jac=True, callback=self._callback,
                          options={
                              'maxiter': self.MaxIterations,
                              'ftol': self.ConvergenceTolerance,
                              'maxcor': 10,  # Number of previous gradients used to approximate the Hessian
                              'disp': True
                          })

        # Write --------------------------------------------------------------------------------------------------------
        self.Write(result.x)

    # def Print(self):
    #     print('')
    #     print('------------------------------------- Iteration: ' + str(self.CurrentIteration)
    #           + ' -------------------------------------')

    def Write(self, x):
        """
        Save the results contained in x.
        """
        fixedEffects = self._unvectorize_fixed_effects(x)
        self.StatisticalModel.set_fixed_effects(fixedEffects)
        self.StatisticalModel.write(self.Dataset)

    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    def _cost_and_derivative(self, x):

        # Recover the structure of the parameters ----------------------------------------------------------------------
        fixed_effects = self._unvectorize_fixed_effects(x)

        # Call the model method ----------------------------------------------------------------------------------------
        attachement, regularity, gradient = self.StatisticalModel.compute_log_likelihood(
            self.Dataset, fixed_effects, None, None, with_grad=True)

        # Prepare the outputs: notably linearize and concatenates the gradient -----------------------------------------
        cost = - attachement - regularity
        gradient = - np.concatenate([value.flatten() for value in gradient.values()])

        return cost, gradient

    def _unvectorize_fixed_effects(self, x):
        """
        Recover the structure of the parameters
        """
        fixed_effects = {}
        cursor = 0
        for key, shape in self.fixed_effects_shape.items():
            length = np.prod(shape)
            fixed_effects[key] = x[cursor:cursor + length].reshape(shape)
            cursor += length
        return fixed_effects

    def _callback(self, x):
        # if not (self.CurrentIteration % self.PrintEveryNIters): self.Print()
        if not (self.CurrentIteration % self.SaveEveryNIters): self.Write(x)

        self.CurrentIteration += 1
