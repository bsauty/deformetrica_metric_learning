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

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def sample(self, statistical_model, dataset, population_RER, individual_RER):

        # Initialization -----------------------------------------------------------------------------------------------
        # Initialization of the memory of the current model terms.
        # The contribution of each subject is stored independently.
        current_model_terms = statistical_model.compute_model_log_likelihood(dataset, population_RER, individual_RER)

        # Main loop ----------------------------------------------------------------------------------------------------
        for random_effect_name, random_effect_distribution in self.individual_proposal_distributions.items():
            # blabla