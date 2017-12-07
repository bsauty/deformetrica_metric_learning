import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

from pydeformetrica.src.core.estimators.abstract_estimator import AbstractEstimator
from torch import optim

class TorchOptimize(AbstractEstimator):

    """
    TorchOptimize object class.
    An estimator is an algorithm which updates the fixed effects of a statistical model.

    """

    ################################################################################
    ### Constructor:
    ################################################################################

    def __init__(self):
        self.CurrentLoss = None

        self.SmallestLoss = None
        self.BestFixedEffects = None


    ################################################################################
    ### Public methods:
    ################################################################################

    # Runs the torch optimize routine and updates the statistical model.
    def Update(self):

        # Initialization -----------------------------------------------------------
        fixedEffects = self.StatisticalModel.GetVectorizedFixedEffects()
        optimizer = optim.Adadelta([fixedEffects], lr=10)

        # Main loop ----------------------------------------------------------------
        for iter in range(1, self.MaxIterations + 1):
            self.CurrentIteration = iter

            # Optimizer step -------------------------------------------------------
            self.CurrentLoss = - self.StatisticalModel.ComputeLogLikelihood(self.Dataset, fixedEffects, None, None)
            self.CurrentLoss.backward()
            optimizer.step()

            # Update memory --------------------------------------------------------
            if ((self.SmallestLoss is None) or (self.CurrentLoss < self.SmallestLoss)):
                self.SmallestLoss = self.CurrentLoss
                self.BestFixedEffects = fixedEffects

            # Printing and writing -------------------------------------------------
            if not(iter % self.PrintEveryNIters):
                self.Print()

            if not(iter % self.WriteEveryNIters):
                self.Write()

        # Finalization -------------------------------------------------------------
        print('Maximum number of iterations is reached.')
        self.Write()

    # Prints information.
    def Print(self):
        print('')
        print('Iteration: ' + str(self.CurrentIteration))
        print('Complete log-likelihood = ' + str(- self.CurrentLoss.data.numpy()[0]))

    # Save the current best results.
    def Write(self):
        self.StatisticalModel.SetFixedEffects(self.BestFixedEffects)
        self.StatisticalModel.Write(self.Dataset)