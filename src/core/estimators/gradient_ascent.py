import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

from pydeformetrica.src.core.estimators.abstract_estimator import AbstractEstimator

import numpy as np
from decimal import Decimal
import torch
import math
import copy


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
        # self.current_fixed_effects = self.statistical_model.get_fixed_effects()
        self.current_parameters = self._get_parameters()

        # self.current_attachment, self.current_regularity, gradient = self.statistical_model.compute_log_likelihood(
        #     self.dataset, self.current_fixed_effects, None, self.individual_RER, with_grad=True)
        self.current_attachment, self.current_regularity, gradient = self._evaluate_model_fit(self.current_parameters,
                                                                                              with_grad=True)
        self.current_log_likelihood = self.current_attachment + self.current_regularity
        self.print()

        initial_log_likelihood = self.current_log_likelihood
        last_log_likelihood = initial_log_likelihood

        nb_params = len(gradient)
        step = np.ones((nb_params,)) * self.initial_step_size

        # Main loop ----------------------------------------------------------------------------------------------------
        for iter in range(1, self.max_iterations + 1):
            self.current_iteration = iter

            # Line search ----------------------------------------------------------------------------------------------
            found_min = False
            for li in range(self.max_line_search_iterations):

                # Print step size --------------------------------------------------------------------------------------
                if not (iter % self.print_every_n_iters):
                    k = 0
                    print('>> Step size = ')
                    for key in gradient.keys():
                        print('\t %.3E [ %s ]' % (Decimal(str(step[k])), key))
                        k += 1

                # Try a simple gradient ascent step --------------------------------------------------------------------
                new_parameters = self._gradient_ascent_step(self.current_parameters, gradient, step)
                # new_attachment, new_regularity = self.statistical_model.compute_log_likelihood(
                #     self.dataset, new_fixed_effects, None, self.individual_RER)
                new_attachment, new_regularity = self._evaluate_model_fit(new_parameters)

                q = new_attachment + new_regularity - last_log_likelihood
                if q > 0:
                    found_min = True
                    step *= self.line_search_expand
                    break

                # Adapting the step sizes ------------------------------------------------------------------------------
                elif nb_params > 1:
                    step *= self.line_search_shrink

                    new_parameters_prop = [None] * nb_params
                    new_attachment_prop = [None] * nb_params
                    new_regularity_prop = [None] * nb_params
                    q_prop = [None] * nb_params

                    for k in range(nb_params):
                        local_step = step
                        local_step[k] /= self.line_search_shrink

                        new_parameters_prop[k] = self._gradient_ascent_step(
                            self.current_parameters, gradient, local_step)
                        # new_attachment_prop[k], new_regularity_prop[k] = self.statistical_model.compute_log_likelihood(
                        #     self.dataset, new_fixed_effects_prop[k], None, self.individual_RER)
                        new_attachment_prop[k], new_regularity_prop[k] = self._evaluate_model_fit(
                            new_parameters_prop[k])

                        q_prop[k] = new_attachment_prop[k] + new_regularity_prop[k] - last_log_likelihood

                    index = q_prop.index(max(q_prop))
                    if q_prop[index] > 0:
                        new_attachment = new_attachment_prop[index]
                        new_regularity = new_regularity_prop[index]
                        new_parameters = new_parameters_prop[index]
                        step[index] /= self.line_search_shrink
                        found_min = True
                        break
                    else:
                        step *= self.line_search_shrink

                else:
                    step *= self.line_search_shrink

            # End of line search ---------------------------------------------------------------------------------------
            if not found_min:
                self.statistical_model.set_fixed_effects(self.current_parameters)
                print('>> Number of line search loops exceeded. Stopping.')
                break

            self.current_attachment = new_attachment
            self.current_regularity = new_regularity
            self.current_log_likelihood = new_attachment + new_regularity
            self.current_parameters = new_parameters

            # Test the stopping criterion ------------------------------------------------------------------------------
            current_log_likelihood = self.current_log_likelihood
            delta_f_current = last_log_likelihood - current_log_likelihood
            delta_f_initial = initial_log_likelihood - current_log_likelihood

            if math.fabs(delta_f_current) < self.convergence_tolerance * math.fabs(delta_f_initial):
                print('>> Tolerance threshold met. Stopping the optimization process.\n')
                break

            # Printing and writing -------------------------------------------------------------------------------------
            if not (iter % self.print_every_n_iters): self.print()
            if not (iter % self.save_every_n_iters): self.write()

            # Prepare next iteration -----------------------------------------------------------------------------------
            last_log_likelihood = current_log_likelihood

            # gradient = self.statistical_model.compute_log_likelihood(
            #     self.dataset, self.current_parameters, None, self.individual_RER, with_grad=True)[2]
            gradient = self._evaluate_model_fit(self.current_parameters, with_grad=True)[2]

        # Finalization -------------------------------------------------------------------------------------------------
        print('>> Write output files ...')
        self.write()
        print('>> Done.')

    def print(self):
        """
        Prints information.
        """
        print('')
        print('------------------------------------- Iteration: ' + str(self.current_iteration)
              + ' -------------------------------------')
        print('>> Log-likelihood = %.3E \t [ attachement = %.3E ; regularity = %.3E ]' %
              (Decimal(str(self.current_log_likelihood)),
               Decimal(str(self.current_attachment)),
               Decimal(str(self.current_regularity))))

    def write(self):
        """
        Save the current results.
        """
        self._update_model(self.current_parameters)
        self.statistical_model.write(self.dataset)

    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    def _get_parameters(self):
        fixed_effects = self.statistical_model.get_fixed_effects()
        out = fixed_effects
        out.update(self.population_RER)
        out.update(self.individual_RER)
        assert len(out) == len(fixed_effects) + len(self.population_RER) + len(self.individual_RER)
        return out

    def _evaluate_model_fit(self, parameters, with_grad=False):
        fixed_effects = {key: parameters[key] for key in self.statistical_model.get_fixed_effects().keys()}
        population_RER = {key: parameters[key] for key in self.population_RER.keys()}
        individual_RER = {key: parameters[key] for key in self.individual_RER.keys()}
        return self.statistical_model.compute_log_likelihood(
            self.dataset, fixed_effects, population_RER, individual_RER, with_grad=with_grad)

    def _gradient_ascent_step(self, parameters, gradient, step):
        new_parameters = copy.deepcopy(parameters)
        for k, key in enumerate(gradient.keys()): new_parameters[key] += gradient[key] * step[k]
        return new_parameters

    def _update_model(self, parameters):
        fixed_effects = {key: parameters[key] for key in self.statistical_model.get_fixed_effects().keys()}
        self.statistical_model.set_fixed_effects(fixed_effects)


