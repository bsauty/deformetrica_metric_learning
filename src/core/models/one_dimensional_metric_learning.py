import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../')

import numpy as np
import math

import torch
from torch.autograd import Variable
#
# from pydeformetrica.src.core.models.abstract_statistical_model import AbstractStatisticalModel
from pydeformetrica.src.support.utilities.general_settings import Settings
from pydeformetrica.src.support.probability_distributions.inverse_wishart_distribution import InverseWishartDistribution
from pydeformetrica.src.support.probability_distributions.multi_scalar_inverse_wishart_distribution import \
    MultiScalarInverseWishartDistribution
from pydeformetrica.src.support.probability_distributions.multi_scalar_normal_distribution import \
    MultiScalarNormalDistribution
# from pydeformetrica.src.support.kernels.kernel_functions import create_kernel
# from pydeformetrica.src.in_out.array_readers_and_writers import *

#To implement:
#-dataset: use of the same class, will put float for the deformable objects.
#-the main geodesic, should be close to a spatiotemporal frame.

class OneDimensionalMetricLearning(AbstractStatisticalModel):
    """
    Deterministic atlas object class.

    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self):
        AbstractStatisticalModel.__init__(self)

        self.initial_cp_spacing = None
        self.number_of_subjects = None
        self.number_of_objects = None

        self.main_geodesic = None

        self.number_interpolation_points = 20

        self.interpolation_points = np.linspace(0., 1., self.number_interpolation_points)
        self.kernel_width = 0.1/self.number_interpolation_points

        # Dictionary of numpy arrays.
        self.fixed_effects['p0'] = None
        self.fixed_effects['reference_time'] = None
        self.fixed_effects['v0'] = None
        self.fixed_effects['onset_age_variance'] = None
        self.fixed_effects['log_acceleration_variance'] = None
        self.fixed_effects['noise_variance'] = None
        self.fixed_effects['metric_parameters'] = np.ones(self.number_interpolation_points-1)/(self.number_interpolation_points)

        # Dictionary of prior distributions
        self.priors['time_shift_variance'] = MultiScalarInverseWishartDistribution()
        self.priors['log_acceleration_variance'] = MultiScalarInverseWishartDistribution()
        self.priors['noise_variance'] = MultiScalarInverseWishartDistribution()

        # Dictionary of probability distributions.
        self.individual_random_effects['onset_age'] = MultiScalarNormalDistribution()
        self.individual_random_effects['log_acceleration'] = MultiScalarNormalDistribution()


    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################

    # Reference time ---------------------------------------------------------------------------------------------------
    def get_reference_time(self):
        return self.fixed_effects['reference_time']

    def set_reference_time(self, rt):
        self.fixed_effects['reference_time'] = rt
        self.individual_random_effects['onset_age'].mean = np.zeros((1,)) + rt

    # Log-acceleration variance ----------------------------------------------------------------------------------------
    def get_log_acceleration_variance(self):
        return self.fixed_effects['log_acceleration_variance']

    def set_log_acceleration_variance(self, lav):
        self.fixed_effects['log_acceleration_variance'] = lav
        self.individual_random_effects['log_acceleration'].set_variance(lav)

    # Time-shift variance ----------------------------------------------------------------------------------------------
    def get_onset_age_variance(self):
        return self.fixed_effects['onset_age_variance']

    def set_onset_age_variance(self, tsv):
        self.fixed_effects['onset_age_variance'] = tsv
        self.individual_random_effects['onset_age'].set_variance(tsv)

    # Noise variance ---------------------------------------------------------------------------------------------------
    def get_noise_variance(self):
        return self.fixed_effects['noise_variance']

    def set_noise_variance(self, nv):
        self.fixed_effects['noise_variance'] = nv

    # Full fixed effects -----------------------------------------------------------------------------------------------
    def get_fixed_effects(self):
        out = {}
        out['p0'] = self.fixed_effects['p0']
        out['v0'] = self.fixed_effects['v0']
        out['metric_parameters'] = self.fixed_effects['metric_parameters']
        return out

    def set_fixed_effects(self, fixed_effects):
        self.fixed_effects['reference_time'] = fixed_effects['reference_time']
        self.fixed_effects['p0'] = fixed_effects['p0']
        self.fixed_effects['v0'] = fixed_effects['v0']
        self.fixed_effects['metric_parameters'] = fixed_effects['metric_parameters']

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def update(self):
        """
        Initializations of prior parameters  + miscellaneous initializations
        """
        pass

    # Compute the functional. Numpy input/outputs.
    def compute_log_likelihood(self, dataset, population_RER, individual_RER, mode='complete', with_grad=False):
        """
        Compute the log-likelihood of the dataset, given parameters fixed_effects and random effects realizations
        population_RER and indRER.

        :param dataset: LongitudinalDataset instance
        :param fixed_effects: Dictionary of fixed effects.
        :param population_RER: Dictionary of population random effects realizations.
        :param indRER: Dictionary of individual random effects realizations.
        :param mode: Indicates which log_likelihood should be computed, between 'complete', 'model', and 'class2'.
        :param with_grad: Flag that indicates wether the gradient should be returned as well.
        :return:
        """
        v0, p0, metric_parameters = self._fixed_effects_to_torch_tensors(with_grad)
        onset_ages, log_accelerations = self._individual_RER_to_torch_tensors(individual_RER, with_grad)

        residuals = self._compute_residuals(dataset, v0, p0, log_accelerations, onset_ages, metric_parameters)

        #Achtung update of the metric parameters
        self.update_fixed_effects(dataset, residuals, individual_RER)
        attachment = self._compute_attachment(residuals)
        regularity = self._compute_random_effects_regularity(log_accelerations, onset_ages)#To implement as well
        regularity += self._compute_class1_priors_regularity()
        regularity += self._compute_class2_priors_regularity()

        if with_grad:
            total = attachment + regularity
            total.backward()

            # Gradients of the effects with no closed form update.
            gradient = {}
            gradient['v0'] = v0.grad.data.cpu().numpy()
            gradient['p0'] = p0.grad.data.cpu().numpy()
            gradient['metric_parameters'] = metric_parameters.grad.data.cpu().numpy()
            gradient['onset_ages'] = onset_ages.grad.data.cpu().numpy()
            gradient['log_accelerations'] = log_accelerations.grad.data.cpu().numpy()

            return attachment.data.cpu().numpy()[0], regularity.data.cpu().numpy()[0], gradient

        return attachment.data.cpu().numpy()[0], regularity.data.cpu().numpy()[0]


    def _fixed_effects_to_torch_tensors(self, with_grad):
        v0_torch = Variable(torch.from_numpy(np.array([self.fixed_effects['v0']])), requires_grad=with_grad).dtype(Settings().tensor_scalar_type)
        p0_torch = Variable(torch.from_numpy(np.array([self.fixed_effects['p0']])), requires_grad=with_grad).dtype(Settings().tensor_scalar_type)
        return v0_torch, p0_torch

    def _individual_RER_to_torch_tensors(self, individual_RER, with_grad):
        onset_ages = individual_RER['onset_age']
        onset_ages = Variable(torch.from_numpy(onset_ages).type(Settings().tensor_scalar_type),
                              requires_grad=with_grad)
        # Log accelerations.
        log_accelerations = individual_RER['log_acceleration']
        log_accelerations = Variable(torch.from_numpy(log_accelerations).type(Settings().tensor_scalar_type),
                                     requires_grad=with_grad)

        return onset_ages, log_accelerations

    def _compute_residuals(self, dataset, v0, p0, log_accelerations, onset_ages):
        """
        dataset is a list of list !
        """
        targets = dataset.targets #a list of list
        absolute_times = self._compute_absolute_times(dataset.times, log_accelerations, onset_ages)

        residuals = []

        number_of_subjects = dataset.number_of_subjects
        for i in range(number_of_subjects):
            residuals_i = []
            for j, (time, target) in enumerate(zip(absolute_times[i], targets[i])):
                predicted_value = self.main_geodesic.get_value(v0, p0, log_accelerations[i], onset_ages[i], absolute_times[i][j])
                #Target should be a torch tensor here I believe.
                residuals_i.append((target - predicted_value)**2)

        return residuals

    def _compute_attachment(self, residuals):

        number_of_subjects = len(residuals)
        attachments = Variable(torch.zeros((number_of_subjects,)).type(Settings().tensor_scalar_type),
                               requires_grad=False)
        noise_variance_torch = Variable(torch.from_numpy(self.fixed_effects['noise_variance'])
                                        .type(Settings().tensor_scalar_type),
                                        requires_grad=False)

        for i in range(number_of_subjects):
            attachment_i = 0.0
            for j in range(len(residuals[i])):
                attachment_i -= (residuals[i][j] / noise_variance_torch) * 0.5
            attachments[i] = attachment_i
        return torch.sum(attachments)


    def _compute_absolute_times(self, times, onset_ages, log_accelerations):
        """
        Fully torch.
        """
        reference_time = self.get_reference_time()
        accelerations = torch.exp(log_accelerations)

        upper_threshold = math.exp(7.5 * math.sqrt(self.get_log_acceleration_variance()))
        lower_threshold = math.exp(- 7.5 * math.sqrt(self.get_log_acceleration_variance()))
        if np.max(accelerations.data.numpy()) > upper_threshold or np.min(accelerations.data.numpy()) < lower_threshold:
            raise ValueError('Absurd numerical value for the acceleration factor. Exception raised.')

        absolute_times = []
        for i in range(len(times)):
            absolute_times_i = []
            for j in range(len(times[i])):
                absolute_times_i.append(accelerations[i] * (times[i][j] - onset_ages[i]) + reference_time)
            absolute_times.append(absolute_times_i)
        return absolute_times

    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    def _compute_random_effects_regularity(self, sources, onset_ages, log_accelerations):
        """
        Fully torch.
        """
        number_of_subjects = onset_ages.shape[0]
        regularity = 0.0

        # Onset age random effect.
        for i in range(number_of_subjects):
            regularity += self.individual_random_effects['onset_age'].compute_log_likelihood_torch(onset_ages[i])

        # Log-acceleration random effect.
        for i in range(number_of_subjects):
            regularity += \
                self.individual_random_effects['log_acceleration'].compute_log_likelihood_torch(log_accelerations[i])

        # Noise random effect.
        regularity -= 0.5 * number_of_subjects \
                      * math.log(self.fixed_effects['noise_variance'])

        return regularity

    def compute_sufficient_statistics(self, dataset, population_RER, individual_RER, residuals):
        sufficient_statistics = {}

        sufficient_statistics['S1'] = 0.
        for i in range(len(residuals)):
            for j in range(len(residuals[j])):
                sufficient_statistics['S1'] += residuals[i][j]

        log_accelerations = individual_RER['log_accelerations']
        sufficient_statistics['S2'] = np.sum(log_accelerations**2)

        onset_ages = individual_RER['onset_ages']
        sufficient_statistics['S3'] = np.sum(onset_ages)

        sufficient_statistics['S4'] = np.sum((log_accelerations - sufficient_statistics['S3']/len(dataset.targets))**2)

        return sufficient_statistics

    def update_fixed_effects(self, dataset, sufficient_statistics):
        """
        Updates the fixed effects based on the sufficient statistics, maximizing the likelihood.
        """
        number_of_subjects = dataset.number_of_subjects
        total_number_of_observations = dataset.total_number_of_observations


        # Updating the noise variance
        prior_scale = self.priors['noise_variance'].scale_scalars[0]
        prior_dof = self.priors['noise_variance'].degrees_of_freedom[0]
        noise_variance = (sufficient_statistics['S1'] + prior_dof * prior_scale) \
                                    / (total_number_of_observations + prior_dof) # Dimension of objects is 1
        self.set_noise_variance(noise_variance)



        # Updating the log acceleration variance
        prior_scale = self.priors['log_acceleration_variance'].scale_scalars[0]
        prior_dof = self.priors['log_acceleration_variance'].degrees_of_freedom[0]
        log_acceleration_variance = (sufficient_statistics["S2"] + prior_dof * prior_scale) \
                                    / (number_of_subjects + prior_dof)
        self.set_log_acceleration_variance(log_acceleration_variance)


        # Updating the reference time
        reftime = sufficient_statistics['S4']/number_of_subjects
        self.set_reference_time(reftime)

        # Updating the onset ages variance
        onset_age_prior_scale = self.priors['onset_age_variance'].scale_scalars[0]
        onset_age_prior_dof = self.priors['onset_age_variance'].degrees_of_freedom[0]
        onset_age_variance = (sufficient_statistics['S4'] - 2 * reftime * sufficient_statistics['S1']
                               + number_of_subjects * reftime ** 2 + onset_age_prior_dof * onset_age_prior_scale) \
                              / (number_of_subjects + onset_age_prior_scale)
        self.set_onset_age_variance(onset_age_variance)

    def _compute_class1_priors_regularity(self):
        """
        Fully torch.
        Prior terms of the class 1 fixed effects, i.e. those for which we know a close-form update. No derivative
        wrt those fixed effects will therefore be necessary.
        """
        regularity = 0.0

        # Time-shift variance prior
        regularity += \
            self.priors['time_shift_variance'].compute_log_likelihood(self.fixed_effects['time_shift_variance'])

        # Log-acceleration variance prior
        regularity += self.priors['log_acceleration_variance'].compute_log_likelihood(
            self.fixed_effects['log_acceleration_variance'])

        # Noise variance prior
        regularity += self.priors['noise_variance'].compute_log_likelihood(self.fixed_effects['noise_variance'])

        return regularity

    def _compute_class2_priors_regularity(self):
        """
        Fully torch.
        Prior terms of the class 2 fixed effects, i.e. those for which we do not know a close-form update. Derivative
        wrt those fixed effects will therefore be necessary.
        """
        regularity = 0.0
        # We don't have regularity terms on \theta_g, A, v0 or p0.

        return Variable(torch.Tensor([regularity]).type(Settings().tensor_scalar_type))

    ####################################################################################################################
    ### Writing methods:
    ####################################################################################################################

    def write(self, dataset, population_RER, individual_RER):
        pass