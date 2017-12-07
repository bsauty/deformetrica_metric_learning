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

    # def __init__(self):
    #     self.InitialStepSize = None
    #     self.MaxLineSearchIterations = None
    #
    #     self.LineSearchShrink = None
    #     self.LineSearchExpand = None
    #     self.ConvergenceTolerance = None
    #
    #     self.LogLikelihoodHistory = []


    ################################################################################
    ### Public methods:
    ################################################################################

    # Runs the torch optimize routine and updates the statistical model.
    def Update(self):
        fixedEffects = self.StatisticalModel.GetVectorizedFixedEffects()
        optimizer = optim.Adadelta([elt for elt in fixedEffects if elt.requires_grad], lr=10)
        print("Optimizing over :", [elt.size() for elt in fixedEffects if elt.requires_grad])

        for iter in range(self.MaxIterations):
            loss = - self.StatisticalModel.ComputeLogLikelihood(self.Dataset, fixedEffects, None, None)
            loss.backward(retain_graph=True)
            optimizer.step()

            print('')
            print('Iteration: ' + str(iter))
            print('Complete log-likelihood = ' + str(- loss.data.numpy()[0]))

        self.StatisticalModel.SetFixedEffects(fixedEffects)
        self.StatisticalModel.Write(self.Dataset)
