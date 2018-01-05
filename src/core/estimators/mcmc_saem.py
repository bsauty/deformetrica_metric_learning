import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

from pydeformetrica.src.core.estimators.abstract_estimator import AbstractEstimator

import numpy as np
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

        self.sampler = None
        self.sufficient_statistics = None  # Dictionary of numpy arrays.
        self.number_of_burn_in_iterations = None  # Number of iterations without memory.

        self.current_acceptance_rates = {}  # Acceptance rates of the current iteration.
        self.average_acceptance_rates = {}  # Mean acceptance rates, computed over all past iterations.

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

        # Print console information.
        self.print()
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
            self.sufficient_statistics = {key: value + step * (sufficient_statistics[key] - value) for key, value in
                                          self.sufficient_statistics}

            # Maximization and update.
            self.statistical_model.update_fixed_effects(self.dataset, self.sufficient_statistics)

            # Averages the random effect realizations in the concentration phase.
            if step < 1.0:
                scaling_coef_1 = float(iter + 1 - self.number_of_burn_in_iterations)
                scaling_coef_2 = float(iter - self.number_of_burn_in_iterations) / scaling_coef_1
                averaged_population_RER = {key: value * scaling_coef_2 + self.population_RER[key] / scaling_coef_1
                                           for key, value in averaged_population_RER.items()}
                averaged_individual_RER = {key: value * scaling_coef_2 + self.individual_RER[key] / scaling_coef_1
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
        self._set_parameters(self.current_parameters)
        self.statistical_model.write(self.dataset, self.population_RER, self.individual_RER)

    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    def _compute_step_size(self):
        # Blabla
        pass

    def _initialize_number_of_burn_in_iterations(self):
        if self.max_iterations > 4000:
            self.number_of_burn_in_iterations = self.max_iterations - 2000
        else:
            self.number_of_burn_in_iterations = int(self.max_iterations / 2)

    def _initialize_acceptance_rate_information(self):
        self.average_acceptance_rates = {key: 0.0 for key in self.sampler.individual_proposal_distributions.keys()}

    def _update_acceptance_rate_information(self):
        scaling_factor = (float(self.current_iteration) - 1.0) / float(self.current_iteration)
        for key, value in self.average_acceptance_rates.items():
            value = value * scaling_factor + self.current_acceptance_rates[key] / float(self.current_iteration)

    def _initialize_sufficient_statistics(self):
        sufficient_statistics = self.statistical_model.compute_sufficient_statistics(
            self.dataset, self.population_RER, self.individual_RER)
        self.sufficient_statistics = {key: np.zeros(value.shape) for key, value in sufficient_statistics.items()}
        return sufficient_statistics
