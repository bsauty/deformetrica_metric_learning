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
        fixed_effects = self.statistical_model.get_fixed_effects()

        # Dictionary of the shapes of the model parameters.
        self.fixed_effects_shape = {key: value.shape for key, value in fixed_effects.items()}

        # Dictionary of linearized parameters.
        theta = {key: value.flatten() for key, value in fixed_effects.items()}

        # 1D numpy array that concatenates the linearized model parameters.
        x0 = np.concatenate([value for value in theta.values()])

        # Main loop ----------------------------------------------------------------------------------------------------
        self.current_iteration = 1
        print('')

        result = minimize(self._cost_and_derivative, x0.astype('float64'),
                          method='L-BFGS-B', jac=True, callback=self._callback,
                          options={
                              'maxiter': self.max_iterations,
                              'ftol': self.convergence_tolerance,
                              'maxcor': 10,  # Number of previous gradients used to approximate the Hessian
                              'disp': True
                          })

        # Write --------------------------------------------------------------------------------------------------------
        self.write(result.x)

    # def Print(self):
    #     print('')
    #     print('------------------------------------- Iteration: ' + str(self.current_iteration)
    #           + ' -------------------------------------')

    def write(self, x):
        """
        Save the results contained in x.
        """
        fixedEffects = self._unvectorize_fixed_effects(x)
        self.statistical_model.set_fixed_effects(fixedEffects)
        self.statistical_model.write(self.dataset)

    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    def _cost_and_derivative(self, x):

        # Recover the structure of the parameters ----------------------------------------------------------------------
        fixed_effects = self._unvectorize_fixed_effects(x)

        # Call the model method ----------------------------------------------------------------------------------------
        attachement, regularity, gradient = self.statistical_model.compute_log_likelihood(
            self.dataset, fixed_effects, None, None, with_grad=True)

        # Prepare the outputs: notably linearize and concatenates the gradient -----------------------------------------
        cost = - attachement - regularity
        gradient = - np.concatenate([value.flatten() for value in gradient.values()])

        return cost.astype('float64'), gradient.astype('float64')

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
        # if not (self.current_iteration % self.print_every_n_iters): self.Print()
        if not (self.current_iteration % self.save_every_n_iters): self.write(x)

        self.current_iteration += 1
