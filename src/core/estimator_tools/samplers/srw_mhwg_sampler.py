import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../../')
from pydeformetrica.src.support.utilities.general_settings import Settings


class SrwMhwgSampler:

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self):
        # Dictionary of probability distributions.
        self.population_proposal_distributions = {}
        self.individual_proposal_distributions = {}