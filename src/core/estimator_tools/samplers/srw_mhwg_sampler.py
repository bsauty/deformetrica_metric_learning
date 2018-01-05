import sys
import os

import numpy as np
import math

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

        # Acceptance rate metrics initialization.
        acceptance_rates = {key: 0.0 for key in self.individual_proposal_distributions.keys()}

        # Main loop ----------------------------------------------------------------------------------------------------
        for random_effect_name, proposal_RED in self.individual_proposal_distributions.items():

            # RED: random effect distribution.
            model_RED = statistical_model.individual_random_effects[random_effect_name]

            # Initialize subject lists.
            current_regularity_terms = []
            candidate_regularity_terms = []
            current_RER = []
            candidate_RER = []

            # Shape parameters of the current random effect realization.
            shape_parameters = individual_RER[random_effect_name][0].shape

            for i in range(dataset.number_of_subjects):

                # Evaluate the current part.
                current_regularity_terms.append(model_RED.compute_log_likelihood(individual_RER[random_effect_name][i]))
                current_RER.append(individual_RER[random_effect_name][i].flatten())

                # Draw the candidate.
                proposal_RED.mean = current_RER[i]
                candidate_RER.append(proposal_RED.sample())

                # Evaluate the candidate part.
                individual_RER[random_effect_name][i] = candidate_RER[i].reshape(shape_parameters)
                candidate_regularity_terms.append(model_RED.compute_log_likelihood(candidate_RER[i]))

            candidate_model_terms = statistical_model.compute_model_log_likelihood(
                dataset, population_RER, individual_RER)

            for i in range(dataset.number_of_subjects):

                # Acceptance rate.
                tau = candidate_model_terms[i] + candidate_regularity_terms[i] \
                      - current_model_terms[i] - current_regularity_terms[i]

                # Reject.
                if math.log(np.random.uniform()) > tau:
                    individual_RER[random_effect_name][i] = current_RER[i].reshape(shape_parameters)

                # Accept.
                else:
                    current_model_terms[i] = candidate_model_terms[i]
                    current_regularity_terms[i] = candidate_regularity_terms[i]
                    acceptance_rates[random_effect_name] += 1.0

            # Acceptance rate final scaling for the considered random effect.
            acceptance_rates[random_effect_name] *= 100.0 / float(dataset.number_of_subjects)

        return acceptance_rates



