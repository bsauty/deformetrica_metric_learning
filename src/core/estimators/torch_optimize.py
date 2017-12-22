import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

import torch
from torch.autograd import Variable
from torch import optim
from decimal import Decimal

from pydeformetrica.src.core.estimators.abstract_estimator import AbstractEstimator


class TorchOptimize(AbstractEstimator):
    """
    TorchOptimize object class.
    An estimator is an algorithm which updates the fixed effects of a statistical model.

    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self):
        AbstractEstimator.__init__(self)

        self.current_attachement = None
        self.current_regularity = None
        self.current_loss = None

        self.smallest_loss = None
        self.best_fixed_effects = None

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def update(self):

        """
        Runs the torch optimize routine and updates the statistical model.

        """

        # Initialization -----------------------------------------------------------------------------------------------
        fixed_effects = self.statistical_model.get_fixed_effects()
        fixed_effects = {key: Variable(torch.from_numpy(value), requires_grad=True)
                        for key, value in fixed_effects.items()}

        optimizer = optim.LBFGS([elt for elt in fixed_effects.values()], max_iter=10, history_size=20)
        # print("Optimizing over :", [elt.size() for elt in fixed_effects.values() if elt.requires_grad])

        # Called at every iteration of the optimizer.
        def closure():
            optimizer.zero_grad()
            self.current_attachement, self.current_regularity = self.statistical_model.compute_log_likelihood_full_torch(
                self.dataset, fixed_effects, None, None)
            self.current_loss = - self.current_attachement - self.current_regularity
            # print(c)
            self.current_loss.backward()
            return self.current_loss

        # Main loop ----------------------------------------------------------------------------------------------------
        for iter in range(1, self.max_iterations + 1):
            self.current_iteration = iter

            # Optimizer step -------------------------------------------------------------------------------------------
            self.current_attachement, self.current_regularity = self.statistical_model.compute_log_likelihood_full_torch(
                self.dataset, fixed_effects, None, None)

            # self.CurrentLoss = - self.CurrentAttachement - self.CurrentRegularity
            # self.CurrentLoss.backward()
            optimizer.step(closure)

            # Update memory --------------------------------------------------------------------------------------------
            if ((self.smallest_loss is None) or (self.current_loss.data.numpy()[0] < self.smallest_loss.data.numpy()[0])):
                self.smallest_loss = self.current_loss
                self.best_fixed_effects = fixed_effects

            # Printing and writing -------------------------------------------------------------------------------------
            if not (iter % self.print_every_n_iters): self.print()
            if not (iter % self.save_every_n_iters): self.write()

        # Finalization -------------------------------------------------------------------------------------------------
        print('')
        print('>> Maximum number of iterations reached. Stopping the optimization process.')
        print('>> Best log-likelihood: %.3E' % (Decimal(str(- self.smallest_loss.data.numpy()[0]))))
        self.write()

    def print(self):
        """
        Prints information.
        """
        print('')
        print('------------------------------------- Iteration: ' + str(self.current_iteration)
              + ' -------------------------------------')
        print('>> Log-likelihood = %.3E \t [ attachement = %.3E ; regularity = %.3E ]' %
              (Decimal(str(- self.current_loss.data.numpy()[0])),
               Decimal(str(self.current_attachement.data.numpy()[0])),
               Decimal(str(self.current_regularity.data.numpy()[0]))))

    def write(self):
        """
        Save the current best results.
        """
        self.statistical_model.set_fixed_effects({key: value.data.numpy()
                                                  for key, value in self.best_fixed_effects.items()})
        self.statistical_model.write(self.dataset)
