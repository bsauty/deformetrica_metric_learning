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

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def update(self):
        """
        Runs the MCMC-SAEM algorithm and updates the statistical model.
        """

        # Initialization -----------------------------------------------------------------------------------------------
        self._initialize_number_of_burn_in_iterations()
        self._initialize_sufficient_statistics()

        # Ensures that all the model fixed effects are initialized.
        sufficient_statistics = self.statistical_model.compute_sufficient_statistics(
            self.dataset, self.population_RER, self.individual_RER)
        self.statistical_model.update_fixed_effects(self.dataset, sufficient_statistics)

        # Print console information.
        self.print()
        print('>> MCMC-SAEM algorithm launched for ' + str(self.max_iterations) + ' iterations ('
              + str(self.number_of_burn_in_iterations) + ' iterations of burn-in).')

        # Initialization of the average random effects realizations.
        averaged_population_RER = np.zeros((self.population_RER.shape))
        averaged_individual_RER = np.zeros((self.individual_RER.shape))

        # Main loop ----------------------------------------------------------------------------------------------------
        for iter in range(1, self.max_iterations + 1):
            self.current_iteration = iter

            # Simulation.
            self.sampler.sample(self.statistical_model, self.dataset, self.population_RER, self.individual_RER)

            # Stochastic approximation.
            sufficient_statistics = self.statistical_model.compute_sufficient_statistics(
                self.dataset, self.population_RER, self.individual_RER)

            step = self._compute_step_size()
            self.sufficient_statistics += step * (sufficient_statistics - self.sufficient_statistics)

            # Maximization and update.
            self.statistical_model.update_fixed_effects(self.dataset, self.sufficient_statistics)

            # Averages the random effect realizations in the concentration phase.
            if step < 1.0:
                scaling_coefficient = float(iter - self.number_of_burn_in_iterations) \
                                      / float(iter + 1 - self.number_of_burn_in_iterations)
                averaged_population_RER *= scaling_coefficient
                averaged_population_RER += self.population_RER / float(iter + 1 - self.number_of_burn_in_iterations)
                averaged_individual_RER *= scaling_coefficient
                averaged_individual_RER += self.individual_RER / float(iter + 1 - self.number_of_burn_in_iterations)

            # Printing and writing.



    def print(self):
        """
        Prints information.
        """
        print('')
        print('------------------------------------- Iteration: ' + str(self.current_iteration)
              + ' -------------------------------------')

    def write(self):
        """
        Save the current results.
        """
        self._set_parameters(self.current_parameters)
        self.statistical_model.write(self.dataset, self.population_RER, self.individual_RER)

    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################


