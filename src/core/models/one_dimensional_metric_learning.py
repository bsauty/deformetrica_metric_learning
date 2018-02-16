import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../')

import numpy as np
import math

import torch
from torch.autograd import Variable

from pydeformetrica.src.in_out.array_readers_and_writers import *
from pydeformetrica.src.core.models.abstract_statistical_model import AbstractStatisticalModel
from pydeformetrica.src.support.utilities.general_settings import Settings
from pydeformetrica.src.support.probability_distributions.inverse_wishart_distribution import InverseWishartDistribution
from pydeformetrica.src.support.probability_distributions.multi_scalar_inverse_wishart_distribution import \
    MultiScalarInverseWishartDistribution
from pydeformetrica.src.support.probability_distributions.multi_scalar_normal_distribution import \
    MultiScalarNormalDistribution
import matplotlib.pyplot as plt
from matplotlib.colors import cnames

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

        self.geodesic = None

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
        self.fixed_effects['metric_parameters'] = None

        # Dictionary of prior distributions
        self.priors['onset_age_variance'] = MultiScalarInverseWishartDistribution()
        self.priors['log_acceleration_variance'] = MultiScalarInverseWishartDistribution()
        self.priors['noise_variance'] = MultiScalarInverseWishartDistribution()

        # Dictionary of probability distributions.
        self.individual_random_effects['onset_age'] = MultiScalarNormalDistribution()
        self.individual_random_effects['log_acceleration'] = MultiScalarNormalDistribution()

        self.is_frozen = {}
        self.is_frozen['v0'] = True
        self.is_frozen['p0'] = True
        self.is_frozen['reference_time'] = False
        self.is_frozen['onset_age_variance'] = False
        self.is_frozen['log_acceleration_variance'] = False
        self.is_frozen['noise_variance'] = False
        self.is_frozen['metric_parameters'] = True

    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################

    def set_v0(self, v0):
        aux = np.array([v0])
        self.fixed_effects['v0'] = np.array([v0]).flatten()

    def set_p0(self, p0):
        self.fixed_effects['p0'] = np.array([p0]).flatten()

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

    def get_metric_parameters(self):
        return self.fixed_effects['metric_parameters']

    def set_metric_parameters(self, metric_parameters):
        assert abs(np.sum(metric_parameters) - 1) < 1e-5, "Metric not properly normalized"
        self.fixed_effects['metric_parameters'] = metric_parameters

    # Full fixed effects -----------------------------------------------------------------------------------------------
    def get_fixed_effects(self):
        out = {}
        if not self.is_frozen['p0']: out['p0'] = np.array([self.fixed_effects['p0']])
        if not self.is_frozen['v0']: out['v0'] = np.array([self.fixed_effects['v0']])
        if not self.is_frozen['metric_parameters']: out['metric_parameters'] = self.fixed_effects['metric_parameters']
        return out

    def set_fixed_effects(self, fixed_effects):
        if not self.is_frozen['p0']: self.set_p0(fixed_effects['p0'])
        if not self.is_frozen['v0']: self.set_v0(fixed_effects['v0'])
        if not self.is_frozen['metric_parameters']: self.set_metric_parameters(fixed_effects['metric_parameters'])
        for key in fixed_effects.keys():
            print(key, self.fixed_effects[key])

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def update(self):
        """
        Initializations of prior parameters  + miscellaneous initializations
        """
        self.initialize_noise_variables()
        self.initialize_onset_age_variables()
        self.initialize_log_acceleration_variables()


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
        # log_accelerations_frozen = Variable(torch.from_numpy(np.zeros((dataset.number_of_subjects,)))).type(Settings().tensor_scalar_type)

        residuals = self._compute_residuals(dataset, v0, p0, log_accelerations, onset_ages, metric_parameters)


        sufficient_statistics = self.compute_sufficient_statistics(dataset, population_RER, individual_RER, residuals)

        #Achtung update of the metric parameters
        self.update_fixed_effects(dataset, sufficient_statistics)
        attachment = self._compute_attachment(residuals)
        regularity = self._compute_random_effects_regularity(log_accelerations, onset_ages)#To implement as well
        regularity += self._compute_class1_priors_regularity()
        regularity += self._compute_class2_priors_regularity()

        # regularity = Variable(torch.Tensor([0.])).type(Settings().tensor_scalar_type)

        if with_grad:
            total = attachment + regularity
            total.backward(retain_graph=True)

            # Gradients of the effects with no closed form update.
            gradient = {}
            if not self.is_frozen['v0']: gradient['v0'] = v0.grad.data.cpu().numpy()
            if not self.is_frozen['p0']: gradient['p0'] = p0.grad.data.cpu().numpy()
            if not self.is_frozen['metric_parameters']:
                gradient['metric_parameters'] = metric_parameters.grad.data.cpu().numpy()
                #We project the gradient of the metric parameters onto the orthogonal of the constraint.
                orthogonal_gradient = np.ones(len(gradient['metric_parameters']))
                orthogonal_gradient /= np.linalg.norm(orthogonal_gradient)
                gradient['metric_parameters'] -= np.dot(gradient['metric_parameters'], orthogonal_gradient) * orthogonal_gradient
                sp = abs(np.dot(gradient['metric_parameters'], orthogonal_gradient))
                assert sp < 1e-10, "Gradient incorrectly projected %f" %sp

            gradient['onset_age'] = onset_ages.grad.data.cpu().numpy()
            gradient['log_acceleration'] = log_accelerations.grad.data.cpu().numpy()

            for key in gradient.keys():
                print("Gradient norm    ", key, np.linalg.norm(gradient[key]))

            return attachment.data.cpu().numpy()[0], regularity.data.cpu().numpy()[0], gradient

        return attachment.data.cpu().numpy()[0], regularity.data.cpu().numpy()[0]


    def _fixed_effects_to_torch_tensors(self, with_grad):
        v0_torch = Variable(torch.from_numpy(self.fixed_effects['v0']),
                            requires_grad=((not self.is_frozen['v0']) and with_grad))\
            .type(Settings().tensor_scalar_type)

        p0_torch = Variable(torch.from_numpy(self.fixed_effects['p0']),
                            requires_grad=((not self.is_frozen['p0']) and with_grad))\
            .type(Settings().tensor_scalar_type)

        metric_parameters = Variable(torch.from_numpy(
            self.fixed_effects['metric_parameters']), requires_grad=((not self.is_frozen['metric_parameters']) and with_grad)).type(Settings().tensor_scalar_type)
        return v0_torch, p0_torch, metric_parameters

    def _individual_RER_to_torch_tensors(self, individual_RER, with_grad):
        onset_ages = individual_RER['onset_age']
        onset_ages = Variable(torch.from_numpy(onset_ages).type(Settings().tensor_scalar_type),
                              requires_grad=with_grad)
        # Log accelerations.
        log_accelerations = individual_RER['log_acceleration']
        log_accelerations = Variable(torch.from_numpy(log_accelerations).type(Settings().tensor_scalar_type),
                                     requires_grad=with_grad)

        return onset_ages, log_accelerations

    def _compute_residuals(self, dataset, v0, p0, log_accelerations, onset_ages, metric_parameters):
        """
        dataset is a list of list !
        """
        targets = dataset.deformable_objects # A list of list
        absolute_times = self._compute_absolute_times(dataset.times, log_accelerations, onset_ages)

        residuals = []

        t0 = self.get_reference_time()

        self.geodesic.set_t0(t0)
        self.geodesic.set_position_t0(p0)
        self.geodesic.set_velocity_t0(v0)
        self.geodesic.set_tmin(min([subject_times[0].data.numpy()[0]
                                                          for subject_times in absolute_times] + [t0]))
        self.geodesic.set_tmax(max([subject_times[-1].data.numpy()[0]
                                                          for subject_times in absolute_times] + [t0]))
        self.geodesic.set_parameters(metric_parameters)

        self.geodesic.update()

        number_of_subjects = dataset.number_of_subjects
        for i in range(number_of_subjects):
            residuals_i = []
            for j, (time, target) in enumerate(zip(absolute_times[i], targets[i])):
                predicted_value = self.geodesic.get_geodesic_point(absolute_times[i][j])
                #Target should be a torch tensor here I believe.
                residuals_i.append((target - predicted_value)**2)
            residuals.append(residuals_i)

        return residuals

    def _compute_attachment(self, residuals):

        total_residual = 0

        number_of_subjects = len(residuals)
        attachments = Variable(torch.zeros((number_of_subjects,)).type(Settings().tensor_scalar_type),
                               requires_grad=False)
        noise_variance_torch = Variable(torch.from_numpy(np.array([self.fixed_effects['noise_variance']]))
                                        .type(Settings().tensor_scalar_type),
                                        requires_grad=False)

        for i in range(number_of_subjects):
            attachment_i = 0.0
            for j in range(len(residuals[i])):
                attachment_i -= (residuals[i][j] / noise_variance_torch) * 0.5
                # attachment_i -= (residuals[i][j]) * 0.5
                total_residual += residuals[i][j]
            attachments[i] = attachment_i

        print("Residuals :", total_residual.data.numpy())
        return torch.sum(attachments)


    def _compute_absolute_times(self, times, log_accelerations, onset_ages):
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
                absolute_times_i.append(self._compute_absolute_time(times[i][j], accelerations[i], onset_ages[i], reference_time))
            absolute_times.append(absolute_times_i)
        return absolute_times

    def _compute_absolute_time(self, time, acceleration, onset_age, reference_time):
        return acceleration * (time - onset_age) + reference_time

    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    def _compute_random_effects_regularity(self, log_accelerations, onset_ages):
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

        if not self.is_frozen['noise_variance']:
            sufficient_statistics['S1'] = 0.
            for i in range(len(residuals)):
                for j in range(len(residuals[i])):
                    sufficient_statistics['S1'] += residuals[i][j].data.numpy()[0]

        if not self.is_frozen['log_acceleration_variance']:
            log_accelerations = individual_RER['log_acceleration']
            sufficient_statistics['S2'] = np.sum(log_accelerations**2)

            onset_ages = individual_RER['onset_age']
            sufficient_statistics['S3'] = np.sum(onset_ages)

        if not self.is_frozen['onset_age_variance']:
            ref_time = sufficient_statistics['S3']/dataset.number_of_subjects
            sufficient_statistics['S4'] = np.sum((log_accelerations - ref_time)**2)

        return sufficient_statistics

    def update_fixed_effects(self, dataset, sufficient_statistics):
        """
        Updates the fixed effects based on the sufficient statistics, maximizing the likelihood.
        """
        number_of_subjects = dataset.number_of_subjects
        total_number_of_observations = dataset.total_number_of_observations

        # Updating the noise variance
        if not self.is_frozen['noise_variance']:
            prior_scale = self.priors['noise_variance'].scale_scalars[0]
            prior_dof = self.priors['noise_variance'].degrees_of_freedom[0]
            noise_variance = (sufficient_statistics['S1'] + prior_dof * prior_scale) \
                                        / (total_number_of_observations + prior_dof) # Dimension of objects is 1
            self.set_noise_variance(noise_variance)

        # Updating the log acceleration variance
        if not self.is_frozen['log_acceleration_variance']:
            prior_scale = self.priors['log_acceleration_variance'].scale_scalars[0]
            prior_dof = self.priors['log_acceleration_variance'].degrees_of_freedom[0]
            log_acceleration_variance = (sufficient_statistics["S2"] + prior_dof * prior_scale) \
                                        / (number_of_subjects + prior_dof)
            self.set_log_acceleration_variance(log_acceleration_variance)

        # Updating the reference time
        if not self.is_frozen['reference_time']:
            reftime = sufficient_statistics['S3']/number_of_subjects
            self.set_reference_time(reftime)

        # Updating the onset ages variance
        if not self.is_frozen['onset_age_variance']:
            reftime = self.get_reference_time()
            onset_age_prior_scale = self.priors['onset_age_variance'].scale_scalars[0]
            onset_age_prior_dof = self.priors['onset_age_variance'].degrees_of_freedom[0]
            onset_age_variance = (sufficient_statistics['S4'] + onset_age_prior_dof * onset_age_prior_scale) \
                                  / (number_of_subjects + onset_age_prior_scale)
            self.set_onset_age_variance(onset_age_variance)

        print("Log acceleration std", math.sqrt(self.get_log_acceleration_variance()),
              "Onset age std", math.sqrt(self.get_onset_age_variance()),
              "Reference time", self.get_reference_time(),
              "Noise variance", self.get_noise_variance())

    def _compute_class1_priors_regularity(self):
        """
        Fully torch.
        Prior terms of the class 1 fixed effects, i.e. those for which we know a close-form update. No derivative
        wrt those fixed effects will therefore be necessary.
        """
        regularity = 0.0

        # Time-shift variance prior
        if not self.is_frozen['onset_age_variance']:
            regularity += \
                self.priors['onset_age_variance'].compute_log_likelihood(self.fixed_effects['onset_age_variance'])

        # Log-acceleration variance prior
        if not self.is_frozen['log_acceleration_variance']:
            regularity += self.priors['log_acceleration_variance'].compute_log_likelihood(
                self.fixed_effects['log_acceleration_variance'])

        # Noise variance prior
        if not self.is_frozen['noise_variance']:
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


    def initialize_noise_variables(self):
        pass

    def initialize_onset_age_variables(self):
        # Check that the onset age random variable mean has been set.
        if self.individual_random_effects['onset_age'].mean is None:
            raise RuntimeError('The set_reference_time method of a LongitudinalAtlas model should be called before '
                               'the update one.')
        # Check that the onset age random variable variance has been set.
        if self.individual_random_effects['onset_age'].variance_sqrt is None:
            raise RuntimeError('The set_time_shift_variance method of a LongitudinalAtlas model should be called '
                               'before the update one.')

        if not self.is_frozen['onset_age_variance']:
            # Set the time_shift_variance prior scale to the initial time_shift_variance fixed effect.
            self.priors['onset_age_variance'].scale_scalars.append(self.get_onset_age_variance())
            print('>> The time shift variance prior degrees of freedom parameter is ARBITRARILY set to 1.')
            self.priors['onset_age_variance'].degrees_of_freedom.append(1.0)

    def initialize_log_acceleration_variables(self):
        # Set the log_acceleration random variable mean.
        self.individual_random_effects['log_acceleration'].mean = np.zeros((1,))
        # Set the log_acceleration_variance fixed effect.
        if self.get_log_acceleration_variance() is None:
            print('>> The initial log-acceleration std fixed effect is ARBITRARILY set to 0.5')
            log_acceleration_std = 0.5
            self.set_log_acceleration_variance(log_acceleration_std ** 2)

        if not self.is_frozen["log_acceleration_variance"]:
            # Set the log_acceleration_variance prior scale to the initial log_acceleration_variance fixed effect.
            self.priors['log_acceleration_variance'].scale_scalars.append(self.get_log_acceleration_variance())
            # Arbitrarily set the log_acceleration_variance prior dof to 1.
            print('>> The log-acceleration variance prior degrees of freedom parameter is ARBITRARILY set to 1.')
            self.priors['log_acceleration_variance'].degrees_of_freedom.append(1.0)


    ####################################################################################################################
    ### Writing methods:
    ####################################################################################################################

    def write(self, dataset, population_RER, individual_RER):
        self.geodesic.save_metric_plot()
        self.geodesic.save_geodesic_plot ()
        self._write_individual_RER(dataset, individual_RER)
        self._write_model_predictions(dataset, individual_RER)
        # We need to write p0, v0, t0, the metric parameters
        #The log accelerations
        #The onset_ages
        #Plots of the metric
        #Reconstructed data

    def _write_individual_RER(self, dataset, individual_RER):
        onset_ages = individual_RER['onset_age']
        write_2D_array(onset_ages, "onset_age.txt")
        alphas = np.exp(individual_RER['log_acceleration'])
        write_2D_array(alphas, "alphas.txt")
        write_2D_array(np.array(dataset.subject_ids), "subject_ids.txt")

    def _write_model_predictions(self, dataset, individual_RER):

        v0, p0, metric_parameters = self._fixed_effects_to_torch_tensors(False)
        onset_ages, log_accelerations = self._individual_RER_to_torch_tensors(individual_RER, False)

        accelerations = torch.exp(log_accelerations)

        targets = dataset.deformable_objects  # A list of list
        absolute_times = self._compute_absolute_times(dataset.times, log_accelerations, onset_ages)

        predictions = []

        t0 = self.get_reference_time()

        self.geodesic.set_t0(t0)
        self.geodesic.set_position_t0(p0)
        self.geodesic.set_velocity_t0(v0)
        self.geodesic.set_tmin(min([subject_times[0].data.numpy()[0]
                                    for subject_times in absolute_times] + [t0]))
        self.geodesic.set_tmax(max([subject_times[-1].data.numpy()[0]
                                    for subject_times in absolute_times] + [t0]))
        self.geodesic.set_parameters(metric_parameters)

        self.geodesic.update()

        colors = []
        for name in cnames.keys():
            if len(colors)<12:
                colors.append(name)
            else:
                break

        # colors = cnames.keys()[:20]
        pos = 0
        nb_plot_to_make = 10

        number_of_subjects = dataset.number_of_subjects
        for i in range(number_of_subjects):
            predictions_i = []
            for j, (time, target) in enumerate(zip(absolute_times[i], targets[i])):
                predicted_value = self.geodesic.get_geodesic_point(absolute_times[i][j])
                predictions.append(predicted_value.data.numpy()[0])

            if nb_plot_to_make >0:
                # We also make a plot of the trajectory and save it...
                times_subject = Variable(torch.from_numpy(np.linspace(dataset.times[i][0].data.numpy()[0], dataset.times[i][-1].data.numpy()[0], 100)).type(Settings().tensor_scalar_type))
                absolute_times_subject = [self._compute_absolute_time(t, accelerations[i], onset_ages[i], t0) for t in times_subject]
                trajectory = [self.geodesic.get_geodesic_point(t).data.numpy()[0] for t in absolute_times_subject]
                plt.plot(times_subject.data.numpy(), trajectory, color=colors[pos])

                # Now plotting the real data.
                plt.scatter([t.data.numpy()[0] for t in dataset.times[i]], [t.data.numpy()[0] for t in targets[i]], color=colors[pos])
                pos += 1
                if pos >= len(colors):
                    plt.savefig(os.path.join(Settings().output_dir, "plot_subject_"+str(i-pos)+'_to_'+str(i)+'.pdf'))
                    plt.clf()
                    pos = 0
                    nb_plot_to_make -= 1

        write_2D_array(np.array(predictions), "reconstructed_values.txt")
