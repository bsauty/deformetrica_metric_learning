import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

from pydeformetrica.src.core.estimators.abstract_estimator import AbstractEstimator
from torch import optim
from decimal import Decimal
import time

class TorchOptimize(AbstractEstimator):

    """
    TorchOptimize object class.
    An estimator is an algorithm which updates the fixed effects of a statistical model.

    """

    ################################################################################
    ### Constructor:
    ################################################################################

    def __init__(self):
        self.CurrentAttachement = None
        self.CurrentRegularity = None
        self.CurrentLoss = None

        self.SmallestLoss = None
        self.BestFixedEffects = None


    ################################################################################
    ### Public methods:
    ################################################################################

    def Update(self):

        """
        Runs the torch optimize routine and updates the statistical model.

        """

        # Initialization -----------------------------------------------------------
        fixedEffects = self.StatisticalModel.GetFixedEffects()
        optimizer = optim.LBFGS([elt for elt in fixedEffects.values() if elt.requires_grad], max_iter=10, history_size=20)
        print("Optimizing over :", [elt.size() for elt in fixedEffects.values() if elt.requires_grad])
        startTime = time.time()

        #called at every iteration of the optimizer.
        def closure():
            optimizer.zero_grad()
            self.CurrentAttachement, self.CurrentRegularity = self.StatisticalModel.ComputeLogLikelihood(
                self.Dataset, fixedEffects, None, None)
            self.CurrentLoss = - self.CurrentAttachement - self.CurrentRegularity
            # print(c)
            self.CurrentLoss.backward()
            return self.CurrentLoss


        # Main loop ----------------------------------------------------------------
        for iter in range(1, self.MaxIterations + 1):
            self.CurrentIteration = iter

            # Optimizer step -------------------------------------------------------
            self.CurrentAttachement, self.CurrentRegularity = self.StatisticalModel.ComputeLogLikelihood(
                self.Dataset, fixedEffects, None, None)

            # self.CurrentLoss = - self.CurrentAttachement - self.CurrentRegularity
            # self.CurrentLoss.backward()
            optimizer.step(closure)

            # Update memory --------------------------------------------------------
            if ((self.SmallestLoss is None) or (self.CurrentLoss.data.numpy()[0] < self.SmallestLoss.data.numpy()[0])):
                self.SmallestLoss = self.CurrentLoss
                self.BestFixedEffects = fixedEffects

            # Printing and writing -------------------------------------------------
            if not(iter % self.PrintEveryNIters):
                self.Print()

            if not(iter % self.SaveEveryNIters):
                self.Write()

        # Finalization -------------------------------------------------------------
        print('')
        print('Maximum number of iterations reached. Stopping the optimization process.')
        print('Best log-likelihood: %.3E' % (Decimal(str(- self.SmallestLoss.data.numpy()[0]))))
        self.Write()
        endTime = time.time()
        print("Estimation took", time.strftime("%H:%M:%S", time.gmtime(endTime-startTime)))

    # Prints information.
    def Print(self):
        print('')
        print('------------------------------------- Iteration: ' + str(self.CurrentIteration)
              + ' -------------------------------------')
        print('>> Log-likelihood = %.3E \t [ attachement = %.3E ; regularity = %.3E ]' %
              (Decimal(str(- self.CurrentLoss.data.numpy()[0])),
              Decimal(str(self.CurrentAttachement.data.numpy()[0])),
              Decimal(str(self.CurrentRegularity.data.numpy()[0]))))

    # Save the current best results.
    def Write(self):
        self.StatisticalModel.SetFixedEffects(self.BestFixedEffects)
        self.StatisticalModel.Write(self.Dataset)
