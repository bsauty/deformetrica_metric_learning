import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

import numpy as np
from scipy.optimize import minimize
from decimal import Decimal
import math
import copy

from pydeformetrica.src.core.estimators.abstract_estimator import AbstractEstimator
from pydeformetrica.src.core.estimators.scipy_optimize import ScipyOptimize
from pydeformetrica.src.core.estimators.gradient_ascent import GradientAscent
from src.in_out.array_readers_and_writers import *


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

        self.gradient_based_estimator = None
        self.maximize_every_n_iters = None

        self.sampler = None
        self.sufficient_statistics = None                   # Dictionary of numpy arrays.
        self.number_of_burn_in_iterations = None            # Number of iterations without memory.

        self.current_acceptance_rates = {}                  # Acceptance rates of the current iteration.
        self.average_acceptance_rates = {}                  # Mean acceptance rates, computed over all past iterations.

        self.memory_window_size = 10                        # Size of the averaging window for the acceptance rates.
        self.current_acceptance_rates_in_window = None      # Memory of the last memory_window_size acceptance rates.
        self.average_acceptance_rates_in_window = None      # Moving average of current_acceptance_rates_in_window.

        self.model_parameters_trajectory = None             # Memory of the model parameters along the estimation.
        self.save_model_parameters_every_n_iters = None     # Resolution of the model parameters trajectory.

        self.individual_random_effects_samples_stack = None  # Stack of the last individual random effect samples.

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
        self._initialize_model_parameters_trajectory()
        self._initialize_individual_random_effects_samples_stack()

        # Ensures that all the model fixed effects are initialized.
        self.statistical_model.update_fixed_effects(self.dataset, sufficient_statistics)

        # Print initial console information.
        print('------------------------------------- Iteration: ' + str(self.current_iteration)
              + ' -------------------------------------')
        print('>> MCMC-SAEM algorithm launched for ' + str(self.max_iterations) + ' iterations ('
              + str(self.number_of_burn_in_iterations) + ' iterations of burn-in).')
        self.statistical_model.print(self.individual_RER)

        # Initialization of the average random effects realizations.
        averaged_population_RER = {key: np.zeros(value.shape) for key, value in self.population_RER.items()}
        averaged_individual_RER = {key: np.zeros(value.shape) for key, value in self.individual_RER.items()}

        # Main loop ----------------------------------------------------------------------------------------------------
        while self.current_iteration < self.max_iterations:
            self.current_iteration += 1

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
            if not ((self.current_iteration - 1) % self.maximize_every_n_iters):
                fixed_effects_before_maximization = self.statistical_model.get_fixed_effects()
                self._maximize_over_fixed_effects()
                fixed_effects_after_maximization = self.statistical_model.get_fixed_effects()
                fixed_effects = {key: value + step * (fixed_effects_after_maximization[key] - value)
                                 for key, value in fixed_effects_before_maximization.items()}
                # fixed_effects = fixed_effects_after_maximization
                self.statistical_model.set_fixed_effects(fixed_effects)

            # Averages the random effect realizations in the concentration phase.
            if step < 1.0:
                coefficient_1 = float(self.current_iteration + 1 - self.number_of_burn_in_iterations)
                coefficient_2 = (coefficient_1 - 1.0) / coefficient_1
                averaged_population_RER = {key: value * coefficient_2 + self.population_RER[key] / coefficient_1
                                           for key, value in averaged_population_RER.items()}
                averaged_individual_RER = {key: value * coefficient_2 + self.individual_RER[key] / coefficient_1
                                           for key, value in averaged_individual_RER.items()}
                self._update_individual_random_effects_samples_stack()

            # Printing, writing, saving, adapting.
            if not (self.current_iteration % self.print_every_n_iters): self.print()
            if not (self.current_iteration % self.save_every_n_iters): self.write()
            if not (self.current_iteration % self.save_model_parameters_every_n_iters):
                self._update_model_parameters_trajectory()
            if not (self.current_iteration % self.memory_window_size):
                self.average_acceptance_rates_in_window \
                    = {key: np.mean(self.current_acceptance_rates_in_window[key])
                       for key in self.sampler.individual_proposal_distributions.keys()}
                self.sampler.adapt_proposal_distributions(self.average_acceptance_rates_in_window,
                                                          self.current_iteration,
                                                          not self.current_iteration % self.print_every_n_iters)

        # Finalization -------------------------------------------------------------------------------------------------
        print('>> Write output files ...')
        self.write(averaged_population_RER, averaged_individual_RER)
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
            print('\t\t %.2f \t[ %s ]' % (average_acceptance_rate, random_effect_name))

        # Let the model under optimization print information about itself.
        self.statistical_model.print(self.individual_RER)

    def write(self, population_RER=None, individual_RER=None):
        """
        Save the current results.
        """
        # Call the write method of the statistical model.
        if population_RER is None: population_RER = self.individual_RER
        if individual_RER is None: individual_RER = self.individual_RER
        self.statistical_model.write(self.dataset, self.population_RER, self.individual_RER, update_fixed_effects=False)

        # Save the recorded model parameters trajectory.
        write_2D_array(self.model_parameters_trajectory[
                       0:int(self.current_iteration / float(self.save_model_parameters_every_n_iters)) + 1],
                       self.statistical_model.name + '__EstimatedParameters__Trajectory.txt')

        # Save the memorized individual random effects samples.
        if self.current_iteration > self.number_of_burn_in_iterations:
            write_2D_array(self.individual_random_effects_samples_stack[
                           0:self.current_iteration - self.number_of_burn_in_iterations],
                           self.statistical_model.name + '__EstimatedParameters__IndividualRandomEffectsSamples.txt')

    ####################################################################################################################
    ### Private_maximize_over_remaining_fixed_effects() method and associated utilities:
    ####################################################################################################################

    def _maximize_over_fixed_effects(self):
        """
        Update the model fixed effects for which no closed-form update is available (i.e. based on sufficient
        statistics).
        """

        if self.gradient_based_estimator is None:
            # self.gradient_based_estimator = ScipyOptimize()
            # self.gradient_based_estimator.memory_length = 5

            self.gradient_based_estimator = GradientAscent()
            self.gradient_based_estimator.initial_step_size = 1e-4
            self.gradient_based_estimator.line_search_shrink = 0.5
            self.gradient_based_estimator.line_search_expand = 1.2
            self.gradient_based_estimator.scale_initial_step_size = True

            self.gradient_based_estimator.statistical_model = self.statistical_model
            self.gradient_based_estimator.dataset = self.dataset
            self.gradient_based_estimator.optimized_log_likelihood = 'class2'
            self.gradient_based_estimator.max_iterations = 3
            self.gradient_based_estimator.max_line_search_iterations = 10
            self.gradient_based_estimator.memory_length = 3
            self.gradient_based_estimator.convergence_tolerance = 1e-6
            self.gradient_based_estimator.print_every_n_iters = 1
            self.gradient_based_estimator.save_every_n_iters = 100000

        self.gradient_based_estimator.verbose = not self.current_iteration % self.print_every_n_iters

        if self.gradient_based_estimator.verbose > 0:
            print('')
            print('[ maximizing over the fixed effects with the '
                  + self.gradient_based_estimator.name + ' optimizer ]')
        # else:
        #     print('>> Maximizing over the fixed effects with the '
        #           + self.gradient_based_estimator.name + ' optimizer')

        self.gradient_based_estimator.individual_RER = self.individual_RER
        self.gradient_based_estimator.update()

        if self.gradient_based_estimator.verbose > 0:
            print('')
            print('[ end of the gradient-based maximization ]')

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
        # Initialize average_acceptance_rates.
        self.average_acceptance_rates = {key: 0.0 for key in self.sampler.individual_proposal_distributions.keys()}

        # Initialize current_acceptance_rates_in_window.
        self.current_acceptance_rates_in_window = {key: np.zeros((self.memory_window_size,))
                                                   for key in self.sampler.individual_proposal_distributions.keys()}
        self.average_acceptance_rates_in_window = {key: 0.0
                                                   for key in self.sampler.individual_proposal_distributions.keys()}

    def _update_acceptance_rate_information(self):
        # Update average_acceptance_rates.
        coefficient_1 = float(self.current_iteration)
        coefficient_2 = (coefficient_1 - 1.0) / coefficient_1
        self.average_acceptance_rates = {key: value * coefficient_2 + self.current_acceptance_rates[key] / coefficient_1
                                         for key, value in self.average_acceptance_rates.items()}

        # Update current_acceptance_rates_in_window.
        for key in self.current_acceptance_rates_in_window.keys():
            self.current_acceptance_rates_in_window[key][(self.current_iteration - 1) % self.memory_window_size] \
                = self.current_acceptance_rates[key]

    def _initialize_sufficient_statistics(self):
        sufficient_statistics = self.statistical_model.compute_sufficient_statistics(
            self.dataset, self.population_RER, self.individual_RER)
        self.sufficient_statistics = {key: np.zeros(value.shape) for key, value in sufficient_statistics.items()}
        return sufficient_statistics

    ####################################################################################################################
    ### Model parameters trajectory saving methods:
    ####################################################################################################################

    def _get_vectorized_model_parameters(self):
        return np.concatenate([value.flatten() for value in self.statistical_model.fixed_effects.values()])

    def _initialize_model_parameters_trajectory(self):
        number_of_trajectory_points = 10
        self.save_model_parameters_every_n_iters = int(self.max_iterations / float(number_of_trajectory_points))
        if self.save_model_parameters_every_n_iters == 0: self.save_model_parameters_every_n_iters = 1
        x = self._get_vectorized_model_parameters()
        self.model_parameters_trajectory = np.zeros((number_of_trajectory_points + 1, x.size))
        self.model_parameters_trajectory[0] = x

    def _update_model_parameters_trajectory(self):
        self.model_parameters_trajectory[
            int(self.current_iteration / float(self.save_model_parameters_every_n_iters))] \
            = self._get_vectorized_model_parameters()

    def _get_vectorized_individual_RER(self):
        return np.concatenate([value.flatten() for value in self.individual_RER.values()])

    def _initialize_individual_random_effects_samples_stack(self):
        number_of_concentration_iterations = self.max_iterations - self.number_of_burn_in_iterations
        x = self._get_vectorized_individual_RER()
        self.individual_random_effects_samples_stack = np.zeros((number_of_concentration_iterations, x.size))

    def _update_individual_random_effects_samples_stack(self):
        self.individual_random_effects_samples_stack[
            self.current_iteration - self.number_of_burn_in_iterations - 1] = self._get_vectorized_individual_RER()
