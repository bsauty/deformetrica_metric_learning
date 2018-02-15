import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

import numpy as np
from decimal import Decimal
import math
import copy
import _pickle as pickle

from pydeformetrica.src.core.estimators.abstract_estimator import AbstractEstimator
from pydeformetrica.src.support.utilities.general_settings import Settings


class GradientAscent(AbstractEstimator):
    """
    GradientAscent object class.
    An estimator is an algorithm which updates the fixed effects of a statistical model.

    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self):
        AbstractEstimator.__init__(self)
        self.name = 'GradientAscent'

        self.current_parameters = None
        self.current_attachment = None
        self.current_regularity = None
        self.current_log_likelihood = None

        self.initial_step_size = 1.
        self.max_line_search_iterations = 10

        self.line_search_shrink = None
        self.line_search_expand = None
        self.convergence_tolerance = 0.001

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def update(self):

        """
        Runs the gradient ascent algorithm and updates the statistical model.

        """

        # Initialisation -----------------------------------------------------------------------------------------------
        # First case: we use the initialization stored in the state file
        if Settings().load_state:
            self.current_parameters, self.current_iteration = self._load_state_file()
            self._set_parameters(self.current_parameters)  # Propagate the parameter values.
            print("State file loaded, it was at iteration", self.current_iteration)

        # Second case: we use the native initialization of the model.
        else:
            self.current_parameters = self._get_parameters()
            self.current_iteration = 0

        self.current_attachment, self.current_regularity, gradient = self._evaluate_model_fit(self.current_parameters,
                                                                                              with_grad=True)
        self.current_log_likelihood = self.current_attachment + self.current_regularity
        self.print()

        initial_log_likelihood = self.current_log_likelihood
        last_log_likelihood = initial_log_likelihood

        nb_params = len(gradient)
        step = self._initialize_step_size(gradient.keys())

        # Main loop ----------------------------------------------------------------------------------------------------
        while self.current_iteration < self.max_iterations:
            self.current_iteration += 1

            # Line search ----------------------------------------------------------------------------------------------
            found_min = False
            for li in range(self.max_line_search_iterations):

                # Print step size --------------------------------------------------------------------------------------
                if not (self.current_iteration % self.print_every_n_iters):
                    print('>> Step size = ')
                    for key in gradient.keys(): print('\t %.3E [ %s ]' % (Decimal(str(step[key])), key))

                # Try a simple gradient ascent step --------------------------------------------------------------------
                new_parameters = self._gradient_ascent_step(self.current_parameters, gradient, step)
                new_attachment, new_regularity = self._evaluate_model_fit(new_parameters)

                q = new_attachment + new_regularity - last_log_likelihood
                if q > 0:
                    found_min = True
                    step = {key: value * self.line_search_expand for key, value in step.items()}
                    break

                # Adapting the step sizes ------------------------------------------------------------------------------
                step = {key: value * self.line_search_shrink for key, value in step.items()}
                if nb_params > 1:
                    new_parameters_prop = {}
                    new_attachment_prop = {}
                    new_regularity_prop = {}
                    q_prop = {}

                    for key in step.keys():
                        local_step = step.copy()
                        local_step[key] /= self.line_search_shrink

                        new_parameters_prop[key] = self._gradient_ascent_step(self.current_parameters, gradient,
                                                                              local_step)
                        new_attachment_prop[key], new_regularity_prop[key] = self._evaluate_model_fit(
                            new_parameters_prop[key])

                        q_prop[key] = new_attachment_prop[key] + new_regularity_prop[key] - last_log_likelihood

                    key_max = max(q_prop.keys(), key=(lambda key: q_prop[key]))
                    if q_prop[key_max] > 0:
                        new_attachment = new_attachment_prop[key_max]
                        new_regularity = new_regularity_prop[key_max]
                        new_parameters = new_parameters_prop[key_max]
                        step[key_max] /= self.line_search_shrink
                        found_min = True
                        break

            # End of line search ---------------------------------------------------------------------------------------
            if not found_min:
                self._set_parameters(self.current_parameters)
                print('>> Number of line search loops exceeded. Stopping.')
                break

            self.current_attachment = new_attachment
            self.current_regularity = new_regularity
            self.current_log_likelihood = new_attachment + new_regularity
            self.current_parameters = new_parameters
            self._set_parameters(self.current_parameters)

            # Test the stopping criterion ------------------------------------------------------------------------------
            current_log_likelihood = self.current_log_likelihood
            delta_f_current = last_log_likelihood - current_log_likelihood
            delta_f_initial = initial_log_likelihood - current_log_likelihood

            if math.fabs(delta_f_current) < self.convergence_tolerance * math.fabs(delta_f_initial):
                print('>> Tolerance threshold met. Stopping the optimization process.')
                break

            # Printing and writing -------------------------------------------------------------------------------------
            if not self.current_iteration % self.print_every_n_iters: self.print()
            if not self.current_iteration % self.save_every_n_iters: self.write()

            # Prepare next iteration -----------------------------------------------------------------------------------
            last_log_likelihood = current_log_likelihood
            gradient = self._evaluate_model_fit(self.current_parameters, with_grad=True)[2]

            # Save the state.
            if not self.current_iteration % self.save_every_n_iters: self._dump_state_file()

    def print(self):
        """
        Prints information.
        """
        print('')
        print('------------------------------------- Iteration: ' + str(self.current_iteration)
              + ' -------------------------------------')
        print('>> Log-likelihood = %.3E \t [ attachment = %.3E ; regularity = %.3E ]' %
              (Decimal(str(self.current_log_likelihood)),
               Decimal(str(self.current_attachment)),
               Decimal(str(self.current_regularity))))

    def write(self):
        """
        Save the current results.
        """
        self.statistical_model.write(self.dataset, self.population_RER, self.individual_RER)

    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    def _initialize_step_size(self, gradient_keys):
        fixed_effects_keys = self.statistical_model.get_fixed_effects().keys()
        step = {key: 0.0 for key in gradient_keys}
        for key in gradient_keys:
            if key in fixed_effects_keys:
                step[key] = self.initial_step_size
            elif key == 'onset_age':
                # step[key] = 10.0 * self.initial_step_size
                step[key] = self.initial_step_size
            else:
                # step[key] = 10.0 * self.initial_step_size
                step[key] = self.initial_step_size
        return step

    def _get_parameters(self):
        out = self.statistical_model.get_fixed_effects()
        out.update(self.population_RER)
        out.update(self.individual_RER)
        assert len(out) == len(self.statistical_model.get_fixed_effects()) \
                           + len(self.population_RER) + len(self.individual_RER)
        return out

    def _evaluate_model_fit(self, parameters, with_grad=False):
        # Propagates the parameter value to all necessary attributes.
        self._set_parameters(parameters)

        # Call the model method.
        try:
            return self.statistical_model.compute_log_likelihood(
                self.dataset, self.population_RER, self.individual_RER,
                mode=self.optimized_log_likelihood, with_grad=with_grad)

        except ValueError as error:
            print('>> ' + str(error) + ' [ in gradient_ascent ]')
            return - float('inf'), - float('inf')

    def _gradient_ascent_step(self, parameters, gradient, step):
        new_parameters = copy.deepcopy(parameters)
        for key in gradient.keys(): new_parameters[key] += gradient[key] * step[key]
        return new_parameters

    def _set_parameters(self, parameters):
        fixed_effects = {key: parameters[key] for key in self.statistical_model.get_fixed_effects().keys()}
        self.statistical_model.set_fixed_effects(fixed_effects)
        self.population_RER = {key: parameters[key] for key in self.population_RER.keys()}
        self.individual_RER = {key: parameters[key] for key in self.individual_RER.keys()}

    def _load_state_file(self):
        d = pickle.load(open(Settings().state_file, 'rb'))
        return d['current_parameters'], d['current_iteration']

    def _dump_state_file(self):
        d = {'current_parameters': self.current_parameters, 'current_iteration': self.current_iteration}
        pickle.dump(d, open(Settings().state_file, 'wb'))
