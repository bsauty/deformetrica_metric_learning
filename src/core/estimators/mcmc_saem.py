import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

from pydeformetrica.src.core.estimators.abstract_estimator import AbstractEstimator

import numpy as np
from scipy.optimize import minimize
from decimal import Decimal
import math
import copy


class McmcSaem(AbstractEstimator):
    """
    GradientAscent object class.
    An estimator is an algorithm which updates the fixed effects of a statistical model.

    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self):
        AbstractEstimator.__init__(self)
        self.name = 'McmcSaem'

        self.sampler = None
        self.sufficient_statistics = None  # Dictionary of numpy arrays.
        self.number_of_burn_in_iterations = None  # Number of iterations without memory.

        self.current_acceptance_rates = {}  # Acceptance rates of the current iteration.
        self.average_acceptance_rates = {}  # Mean acceptance rates, computed over all past iterations.

        self.fixed_effects_shape = None

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def update(self):
        """
        Runs the MCMC-SAEM algorithm and updates the statistical model.
        """

        # Initialization -----------------------------------------------------------------------------------------------
        self._initialize_number_of_burn_in_iterations()
        self._initialize_acceptance_rate_information()
        sufficient_statistics = self._initialize_sufficient_statistics()

        # Ensures that all the model fixed effects are initialized.
        self.statistical_model.update_fixed_effects(self.dataset, sufficient_statistics)

        # Print initial console information.
        print('------------------------------------- Iteration: ' + str(self.current_iteration)
              + ' -------------------------------------')
        print('>> MCMC-SAEM algorithm launched for ' + str(self.max_iterations) + ' iterations ('
              + str(self.number_of_burn_in_iterations) + ' iterations of burn-in).')

        # Initialization of the average random effects realizations.
        averaged_population_RER = {key: np.zeros(value.shape) for key, value in self.population_RER.items()}
        averaged_individual_RER = {key: np.zeros(value.shape) for key, value in self.individual_RER.items()}

        # Main loop ----------------------------------------------------------------------------------------------------
        for iter in range(1, self.max_iterations + 1):
            self.current_iteration = iter

            # Simulation.
            self.current_acceptance_rates = self.sampler.sample(self.statistical_model, self.dataset,
                                                                self.population_RER, self.individual_RER)
            self._update_acceptance_rate_information()

            # Stochastic approximation.
            sufficient_statistics = self.statistical_model.compute_sufficient_statistics(
                self.dataset, self.population_RER, self.individual_RER)

            step = self._compute_step_size()
            self.sufficient_statistics = {key: value + step * (sufficient_statistics[key] - value)
                                          for key, value in self.sufficient_statistics.items()}

            # Maximization.
            self.statistical_model.update_fixed_effects(self.dataset, self.sufficient_statistics)
            if not (self.current_iteration % 50): self._maximize_over_fixed_effects()

            # Averages the random effect realizations in the concentration phase.
            if step < 1.0:
                coefficient_1 = float(iter + 1 - self.number_of_burn_in_iterations)
                coefficient_2 = (coefficient_1 - 1.0) / coefficient_1
                averaged_population_RER = {key: value * coefficient_2 + self.population_RER[key] / coefficient_1
                                           for key, value in averaged_population_RER.items()}
                averaged_individual_RER = {key: value * coefficient_2 + self.individual_RER[key] / coefficient_1
                                           for key, value in averaged_individual_RER.items()}

            # Printing and writing.
            if not (self.current_iteration % self.print_every_n_iters): self.print()
            if not (self.current_iteration % self.save_every_n_iters): self.write()

        # Finalization -------------------------------------------------------------------------------------------------
        print('>> Write output files ...')
        self.write()
        print('>> Done.')

    def print(self):
        """
        Prints information.
        """
        # Iteration number.
        print('')
        print('------------------------------------- Iteration: ' + str(self.current_iteration)
              + ' -------------------------------------')
        # Averaged acceptance rates over all the past iterations.
        print('>> Average acceptance rates (all past iterations):')
        for random_effect_name, average_acceptance_rate in self.average_acceptance_rates.items():
            print('\t\t %.2f [ %s ]' % (average_acceptance_rate, random_effect_name))

    def write(self):
        """
        Save the current results.
        """
        self.statistical_model.write(self.dataset, self.population_RER, self.individual_RER)

    ####################################################################################################################
    ### Private_maximize_over_remaining_fixed_effects() method and associated utilities:
    ####################################################################################################################

    def _maximize_over_fixed_effects(self):
        """
        Update the model fixed effects for which no closed-form update is available (i.e. based on sufficient
        statistics).
        """
        fixed_effects = self.statistical_model.get_fixed_effects()
        if len(fixed_effects) > 0:
            self.fixed_effects_shape = {key: value.shape for key, value in fixed_effects.items()}
            x0 = np.concatenate([value.flatten() for value in fixed_effects.values()])
            result = minimize(self._cost_and_derivative, x0.astype('float64'), method='L-BFGS-B', jac=True,
                              options={
                                  'maxiter': 5 - 2,  # No idea why the '-2' is necessary.
                                  'ftol': 1e-4,
                                  'maxcor': 5,  # Number of previous gradients used to approximate the Hessian.
                                  'disp': True,
                              })
            self.statistical_model.set_fixed_effects(self._unvectorize_fixed_effects(result.x))

    def _cost_and_derivative(self, x):
        """
        Compute the cost and associated gradient to be minimized with respect to the fixed effects.
        """
        # Recover the fixed effects structure --------------------------------------------------------------------------
        fixed_effects = self._unvectorize_fixed_effects(x)

        # Call the model method ----------------------------------------------------------------------------------------
        log_likelihood_terms, gradient = self.statistical_model.compute_model_log_likelihood(
            self.dataset, fixed_effects, self.population_RER, self.individual_RER, with_grad=True)

        # Prepare the outputs: notably vectorize and concatenate the gradient ------------------------------------------
        cost = - np.sum(log_likelihood_terms)
        gradient = - np.concatenate([value.flatten() for value in gradient.values()])

        return cost.astype('float64'), gradient.astype('float64')

    def _unvectorize_fixed_effects(self, x):
        """
        Recover the structure of the fixed effects.
        """
        fixed_effects = {}
        cursor = 0
        for key, shape in self.fixed_effects_shape.items():
            length = np.prod(shape)
            fixed_effects[key] = x[cursor:cursor + length].reshape(shape)
            cursor += length
        return fixed_effects

    ####################################################################################################################
    ### Other private methods:
    ####################################################################################################################

    def _compute_step_size(self):
        aux = self.current_iteration - self.number_of_burn_in_iterations + 1
        if aux <= 0:
            return 1.0
        else:
            return aux ** - 0.6

    def _initialize_number_of_burn_in_iterations(self):
        if self.max_iterations > 4000:
            self.number_of_burn_in_iterations = self.max_iterations - 2000
        else:
            self.number_of_burn_in_iterations = int(self.max_iterations / 2)

    def _initialize_acceptance_rate_information(self):
        self.average_acceptance_rates = {key: 0.0 for key in self.sampler.individual_proposal_distributions.keys()}

    def _update_acceptance_rate_information(self):
        coefficient_1 = float(self.current_iteration)
        coefficient_2 = (coefficient_1 - 1.0) / coefficient_1
        self.average_acceptance_rates = {key: value * coefficient_2 + self.current_acceptance_rates[key] / coefficient_1
                                         for key, value in self.average_acceptance_rates.items()}

    def _initialize_sufficient_statistics(self):
        sufficient_statistics = self.statistical_model.compute_sufficient_statistics(
            self.dataset, self.population_RER, self.individual_RER)
        self.sufficient_statistics = {key: np.zeros(value.shape) for key, value in sufficient_statistics.items()}
        return sufficient_statistics
