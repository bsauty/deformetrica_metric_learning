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

        self.current_fixed_effects = None
        self.current_attachement = None
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
        self.current_fixed_effects = self.statistical_model.get_fixed_effects()

        self.current_attachement, self.current_regularity, fixed_effects_grad = self.statistical_model.compute_log_likelihood(
            self.dataset, self.current_fixed_effects, None, None, with_grad=True)
        self.current_log_likelihood = self.current_attachement + self.current_regularity
        self.print()

        initial_log_likelihood = self.current_log_likelihood
        last_log_likelihood = initial_log_likelihood

        nb_params = len(fixed_effects_grad)
        step = np.ones((nb_params,)) * self.initial_step_size

        # Main loop ----------------------------------------------------------------------------------------------------
        for iter in range(1, self.max_iterations + 1):
            self.current_iteration = iter

            # Line search ----------------------------------------------------------------------------------------------
            found_min = False
            for li in range(self.max_line_search_iterations):

                # Print step size --------------------------------------------------------------------------------------
                if not(iter % self.print_every_n_iters):
                    k = 0
                    print('>> Step size = ')
                    for key in fixed_effects_grad.keys():
                        print('\t %.3E [ %s ]' % (Decimal(str(step[k])), key))
                        k += 1

                # Try a simple gradient ascent step --------------------------------------------------------------------
                new_fixed_effects = self.gradient_ascent_step(self.current_fixed_effects, fixed_effects_grad, step)
                new_attachement, new_regularity = self.statistical_model.compute_log_likelihood(
                    self.dataset, new_fixed_effects, None, None)

                q = new_attachement + new_regularity - last_log_likelihood
                if q > 0:
                    found_min = True
                    break

                # Adapting the step sizes ------------------------------------------------------------------------------
                elif nb_params > 1:
                    step *= self.line_search_shrink

                    new_fixed_effects_prop = [None] * nb_params
                    new_attachement_prop = [None] * nb_params
                    new_regularity_prop = [None] * nb_params
                    q_prop = [None] * nb_params

                    for k in range(nb_params):
                        localStep = step
                        localStep[k] /= self.line_search_shrink

                        new_fixed_effects_prop[k] = self.gradient_ascent_step(
                            self.current_fixed_effects, fixed_effects_grad, localStep)
                        new_attachement_prop[k], new_regularity_prop[k] = self.statistical_model.compute_log_likelihood(
                            self.dataset, self.current_fixed_effects, None, None)

                        q_prop[k] = new_attachement_prop[k] + new_regularity_prop[k] - last_log_likelihood

                    index = q_prop.index(max(q_prop))
                    if q_prop[index] > 0:
                        new_attachement = new_attachement_prop[index]
                        new_regularity = new_regularity_prop[index]
                        new_fixed_effects = new_fixed_effects_prop[index]
                        step[index] /= self.line_search_shrink
                        found_min = True
                        break

                else:
                    step *= self.line_search_shrink

            # End of line search ---------------------------------------------------------------------------------------
            if not(found_min):
                self.statistical_model.set_fixed_effects(self.current_fixed_effects)
                print('>> Number of line search loops exceeded. Stopping.')
                break

            self.current_attachement = new_attachement
            self.current_regularity = new_regularity
            self.current_log_likelihood = new_attachement + new_regularity
            self.current_fixed_effects = new_fixed_effects

            # Test the stopping criterion ------------------------------------------------------------------------------
            current_log_likelihood = self.current_log_likelihood
            delta_f_current = last_log_likelihood - current_log_likelihood
            delta_f_initial = initial_log_likelihood - current_log_likelihood

            if math.fabs(delta_f_current) < self.convergence_tolerance * math.fabs(delta_f_initial):
                print('>> Tolerance threshold met. Stopping the optimization process.\n')
                break

            # Printing and writing -------------------------------------------------------------------------------------
            if not(iter % self.print_every_n_iters): self.Print()
            if not(iter % self.save_every_n_iters): self.write()

            # Prepare next iteration -----------------------------------------------------------------------------------
            step *= self.line_search_expand
            last_log_likelihood = current_log_likelihood

            fixed_effects_grad = self.statistical_model.compute_log_likelihood(
                self.dataset, self.current_fixed_effects, None, None, with_grad=True)[2]

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
               Decimal(str(self.current_attachement)),
               Decimal(str(self.current_regularity))))

    def write(self):
        """
        Save the current results.
        """
        self.statistical_model.set_fixed_effects(self.current_fixed_effects)
        self.statistical_model.write(self.dataset)


    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    def gradient_ascent_step(self, fixed_effects, fixed_effects_grad, step):
        new_fixed_effects = copy.deepcopy(fixed_effects)

        for k, key in enumerate(fixed_effects_grad.keys()):
            new_fixed_effects[key] += fixed_effects_grad[key] * step[k]

        return new_fixed_effects