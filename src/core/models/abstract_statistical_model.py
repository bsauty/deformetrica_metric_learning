import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../')

class AbstractStatisticalModel:

    """
    AbstractStatisticalModel object class.
    A statistical model is a generative function, which tries to explain an observed stochastic process.

    """

    ################################################################################
    ### Constructor:
    ################################################################################

    def __init__(self):
        self.name = 'undefined'
        self.fixed_effects = {}
        self.priors = {}
        self.population_random_effects = {}
        self.individual_random_effects = {}