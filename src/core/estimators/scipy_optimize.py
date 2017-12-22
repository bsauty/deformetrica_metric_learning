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
        AbstractEstimator.__init__(self)

        self.memory_length = None
        self.parameters_shape = None

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
        parameters = self._get_parameters()
        self.parameters_shape = {key: value.shape for key, value in parameters.items()}
        x0 = self._vectorize_parameters(parameters)

        # Main loop ----------------------------------------------------------------------------------------------------
        self.current_iteration = 1
        print('')

        result = minimize(self._cost_and_derivative, x0.astype('float64'),
                          method='L-BFGS-B', jac=True, callback=self._callback,
                          options={
                              'maxiter': self.max_iterations - 2,  # No idea why this is necessary.
                              'ftol': self.convergence_tolerance,
                              'maxcor': self.memory_length,  # Number of previous gradients used to approximate the Hessian
                              'disp': True,
                          })

        # Write --------------------------------------------------------------------------------------------------------
        self.write(result.x)

    def write(self, x):
        """
        Save the results contained in x.
        """
        self._set_parameters(self._unvectorize_parameters(x))
        self.statistical_model.write(self.dataset, self.population_RER, self.individual_RER)

    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    def _cost_and_derivative(self, x):

        # Recover the structure of the parameters ----------------------------------------------------------------------
        parameters = self._unvectorize_parameters(x)
        fixed_effects = {key: parameters[key] for key in self.statistical_model.get_fixed_effects().keys()}
        population_RER = {key: parameters[key] for key in self.population_RER.keys()}
        individual_RER = {key: parameters[key] for key in self.individual_RER.keys()}

        # Call the model method ----------------------------------------------------------------------------------------
        attachment, regularity, gradient = self.statistical_model.compute_log_likelihood(
            self.dataset, fixed_effects, population_RER, individual_RER, with_grad=True)

        # Prepare the outputs: notably linearize and concatenates the gradient -----------------------------------------
        cost = - attachment - regularity
        gradient = - np.concatenate([value.flatten() for value in gradient.values()])

        return cost.astype('float64'), gradient.astype('float64')

    def _callback(self, x):
        if not (self.current_iteration % self.save_every_n_iters): self.write(x)
        self.current_iteration += 1

    def _get_parameters(self):
        """
        Return a dictionary of numpy arrays.
        """
        out = self.statistical_model.get_fixed_effects()
        out.update(self.population_RER)
        out.update(self.individual_RER)
        assert len(out) == len(self.statistical_model.get_fixed_effects()) \
                           + len(self.population_RER) + len(self.individual_RER)
        return out

    def _vectorize_parameters(self, parameters):
        """
        Returns a 1D numpy array from a dictionary of numpy arrays.
        """
        return np.concatenate([value.flatten() for value in parameters.values()])

    def _unvectorize_parameters(self, x):
        """
        Recover the structure of the parameters
        """
        parameters = {}
        cursor = 0
        for key, shape in self.parameters_shape.items():
            length = np.prod(shape)
            parameters[key] = x[cursor:cursor + length].reshape(shape)
            cursor += length
        return parameters

    def _set_parameters(self, parameters):
        """
        Updates the model and the random effect realization attributes.
        """
        fixed_effects = {key: parameters[key] for key in self.statistical_model.get_fixed_effects().keys()}
        self.statistical_model.set_fixed_effects(fixed_effects)
        self.population_RER = {key: parameters[key] for key in self.population_RER.keys()}
        self.individual_RER = {key: parameters[key] for key in self.individual_RER.keys()}
