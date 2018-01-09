import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../')

import numpy as np
import math

import torch
from torch.autograd import Variable

from pydeformetrica.src.core.models.abstract_statistical_model import AbstractStatisticalModel
from pydeformetrica.src.in_out.deformable_object_reader import DeformableObjectReader
from pydeformetrica.src.in_out.dataset_functions import create_template_metadata, compute_noise_dimension
from pydeformetrica.src.core.model_tools.deformations.spatiotemporal_reference_frame import SpatiotemporalReferenceFrame
from pydeformetrica.src.core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from pydeformetrica.src.support.utilities.general_settings import Settings
from pydeformetrica.src.core.models.model_functions import create_regular_grid_of_points, compute_sobolev_gradient
from pydeformetrica.src.support.kernels.kernel_functions import create_kernel
from pydeformetrica.src.in_out.utils import *
from pydeformetrica.src.core.model_tools.attachments.multi_object_attachment import MultiObjectAttachment
from pydeformetrica.src.support.probability_distributions.normal_distribution import NormalDistribution
from pydeformetrica.src.support.probability_distributions.inverse_wishart_distribution import InverseWishartDistribution
from pydeformetrica.src.support.probability_distributions.multi_scalar_inverse_wishart_distribution import \
    MultiScalarInverseWishartDistribution
from pydeformetrica.src.support.probability_distributions.multi_scalar_normal_distribution import \
    MultiScalarNormalDistribution


class LongitudinalAtlas(AbstractStatisticalModel):
    """
    Longitudinal atlas object class.
    See "Learning distributions of shape trajectories from longitudinal datasets: a hierarchical model on a manifold
    of diffeomorphisms", Bône et al. (2018), in review.

    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self):
        AbstractStatisticalModel.__init__(self)

        self.template = DeformableMultiObject()
        self.objects_name = []
        self.objects_name_extension = []
        self.objects_noise_dimension = []

        self.multi_object_attachment = None
        self.spatiotemporal_reference_frame = SpatiotemporalReferenceFrame()

        self.use_sobolev_gradient = True
        self.smoothing_kernel_width = None

        self.initial_cp_spacing = None
        self.number_of_objects = None
        self.number_of_control_points = None
        self.bounding_box = None

        # Dictionary of numpy arrays.
        self.fixed_effects['template_data'] = None
        self.fixed_effects['control_points'] = None
        self.fixed_effects['momenta'] = None
        self.fixed_effects['modulation_matrix'] = None
        self.fixed_effects['reference_time'] = None
        self.fixed_effects['time_shift_variance'] = None
        self.fixed_effects['log_acceleration_variance'] = None
        self.fixed_effects['noise_variance'] = None

        # Dictionary of probability distributions.
        self.priors['template_data'] = MultiScalarNormalDistribution()
        self.priors['control_points'] = MultiScalarNormalDistribution()
        self.priors['momenta'] = MultiScalarNormalDistribution()
        self.priors['modulation_matrix'] = MultiScalarNormalDistribution()
        self.priors['reference_time'] = MultiScalarNormalDistribution()
        self.priors['time_shift_variance'] = MultiScalarInverseWishartDistribution()
        self.priors['log_acceleration_variance'] = MultiScalarInverseWishartDistribution()
        self.priors['noise_variance'] = MultiScalarInverseWishartDistribution()

        # Dictionary of probability distributions.
        self.individual_random_effects['sources'] = MultiScalarNormalDistribution()
        self.individual_random_effects['onset_age'] = MultiScalarNormalDistribution()
        self.individual_random_effects['log_acceleration'] = MultiScalarNormalDistribution()

        self.freeze_template = False
        self.freeze_control_points = False

    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################

    # Template data ----------------------------------------------------------------------------------------------------
    def get_template_data(self):
        return self.fixed_effects['template_data']

    def set_template_data(self, td):
        self.fixed_effects['template_data'] = td
        self.template.set_data(td)

    # Control points ---------------------------------------------------------------------------------------------------
    def get_control_points(self):
        return self.fixed_effects['control_points']

    def set_control_points(self, cp):
        self.fixed_effects['control_points'] = cp
        self.number_of_control_points = len(cp)

    # Momenta ----------------------------------------------------------------------------------------------------------
    def get_momenta(self):
        return self.fixed_effects['momenta']

    def set_momenta(self, mom):
        self.fixed_effects['momenta'] = mom

    # Modulation matrix ------------------------------------------------------------------------------------------------
    def get_modulation_matrix(self):
        return self.fixed_effects['modulation_matrix']

    def set_modulation_matrix(self, mm):
        self.fixed_effects['modulation_matrix'] = mm

    # Reference time ---------------------------------------------------------------------------------------------------
    def get_reference_time(self):
        return self.fixed_effects['reference_time']

    def set_reference_time(self, rt):
        self.fixed_effects['reference_time'] = rt
        self.individual_random_effects['onset_age'].mean = np.ones((1,)) * rt

    # Time-shift variance ----------------------------------------------------------------------------------------------
    def get_time_shift_variance(self):
        return self.fixed_effects['time_shift_variance']

    def set_time_shift_variance(self, tsv):
        self.fixed_effects['time_shift_variance'] = tsv
        self.individual_random_effects['onset_age'].set_variance(tsv)

    # Log-acceleration variance ----------------------------------------------------------------------------------------
    def get_log_acceleration_variance(self):
        return self.fixed_effects['log_acceleration_variance']

    def set_log_acceleration_variance(self, lav):
        self.fixed_effects['log_acceleration_variance'] = lav
        self.individual_random_effects['log_acceleration'].set_variance(lav)

    # Noise variance ---------------------------------------------------------------------------------------------------
    def get_noise_variance(self):
        return self.fixed_effects['noise_variance']

    def set_noise_variance(self, nv):
        self.fixed_effects['noise_variance'] = nv

    # Class 2 fixed effects --------------------------------------------------------------------------------------------
    def get_fixed_effects(self):
        out = {}
        if not self.freeze_template: out['template_data'] = self.fixed_effects['template_data']
        if not self.freeze_control_points: out['control_points'] = self.fixed_effects['control_points']
        out['momenta'] = self.fixed_effects['momenta']
        out['modulation_matrix'] = self.fixed_effects['modulation_matrix']
        out['reference_time'] = self.fixed_effects['reference_time']
        return out

    def set_fixed_effects(self, fixed_effects):
        if not self.freeze_template: self.set_template_data(fixed_effects['template_data'])
        if not self.freeze_control_points: self.set_control_points(fixed_effects['control_points'])
        self.set_control_points(fixed_effects['momenta'])
        self.set_control_points(fixed_effects['modulation_matrix'])
        self.set_control_points(fixed_effects['reference_time'])

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def update(self):
        """
        Final initialization steps.
        """

        self.number_of_objects = len(self.template.object_list)
        self.bounding_box = self.template.bounding_box

        self.set_template_data(self.template.get_points())
        if self.fixed_effects['control_points'] is None:
            self._initialize_control_points()
        else:
            self._initialize_bounding_box()
        self._initialize_momenta()
        self._initialize_noise_variance()

        # Compute the functional. Numpy input/outputs.

    def compute_log_likelihood(self, dataset, population_RER, individual_RER, with_grad=False):
        """
        Compute the log-likelihood of the dataset, given parameters fixed_effects and random effects realizations
        population_RER and indRER.
        Start by updating the class 1 fixed effects.

        :param dataset: LongitudinalDataset instance
        :param population_RER: Dictionary of population random effects realizations.
        :param individual_RER: Dictionary of individual random effects realizations.
        :param with_grad: Flag that indicates wether the gradient should be returned as well.
        :return:
        """

        # Initialize: conversion from numpy to torch -------------------------------------------------------------------
        template_data, control_points, momenta, modulation_matrix, reference_time = \
            self._fixed_effects_to_torch_tensors(with_grad)
        sources, onset_ages, log_accelerations = self._individual_RER_to_torch_tensors(individual_RER, with_grad)

        # Deform, update, compute metrics ------------------------------------------------------------------------------
        residuals = self._compute_residuals(dataset, template_data, control_points, momenta, modulation_matrix,
                                            reference_time, sources, onset_ages, log_accelerations)
        sufficient_statistics = self.compute_sufficient_statistics(dataset, population_RER, individual_RER,
                                                                   residuals=residuals)
        self.update_fixed_effects(dataset, sufficient_statistics)
        attachment = self._compute_attachment(residuals)
        regularity = self._compute_random_effects_regularity(momenta)
        regularity += self._compute_priors_regularity(momenta)

        # Compute gradient if needed -----------------------------------------------------------------------------------
        if with_grad:
            total = attachment + regularity
            total.backward()

            gradient = {}
            # Template data.
            if not self.freeze_template:
                if self.use_sobolev_gradient:
                    gradient['template_data'] = compute_sobolev_gradient(
                        template_data.grad, self.smoothing_kernel_width, self.template).data.numpy()
                else:
                    gradient['template_data'] = template_data.grad.data.numpy()
            # Other gradients.
            if not self.freeze_control_points: gradient['control_points'] = control_points.grad.data.numpy()
            gradient['momenta'] = momenta.grad.data.cpu().numpy()
            gradient['modulation_matrix'] = modulation_matrix.grad.data.cpu().numpy()
            gradient['reference_time'] = reference_time.grad.data.cpu().numpy()
            gradient['sources'] = sources.grad.data.cpu().numpy()
            gradient['onset_ages'] = onset_ages.grad.data.cpu().numpy()
            gradient['log_accelerations'] = log_accelerations.grad.data.cpu().numpy()

            return attachment.data.cpu().numpy()[0], regularity.data.cpu().numpy()[0], gradient

        else:
            return attachment.data.cpu().numpy()[0], regularity.data.cpu().numpy()[0]

    def compute_model_log_likelihood(self, dataset, fixed_effects, population_RER, individual_RER, with_grad=False):
        """
        Computes the model log-likelihood, i.e. only the attachment part.
        Returns a list of terms, each element corresponding to a subject.
        Optionally returns the gradient with respect to the non-frozen fixed effects.
        """

        # Initialize: conversion from numpy to torch -------------------------------------------------------------------
        # Template data.
        if not self.freeze_template:
            template_data = fixed_effects['template_data']
            template_data = Variable(torch.from_numpy(template_data).type(Settings().tensor_scalar_type),
                                     requires_grad=with_grad)
        else:
            template_data = self.fixed_effects['template_data']
            template_data = Variable(torch.from_numpy(template_data).type(Settings().tensor_scalar_type),
                                     requires_grad=False)

        # Control points.
        if not self.freeze_control_points:
            control_points = fixed_effects['control_points']
            control_points = Variable(torch.from_numpy(control_points).type(Settings().tensor_scalar_type),
                                      requires_grad=with_grad)
        else:
            control_points = self.fixed_effects['control_points']
            control_points = Variable(torch.from_numpy(control_points).type(Settings().tensor_scalar_type),
                                      requires_grad=False)

        # Momenta.
        momenta = individual_RER['momenta']
        momenta = Variable(torch.from_numpy(momenta).type(Settings().tensor_scalar_type), requires_grad=False)

        # Compute residual, and then the attachment term ---------------------------------------------------------------
        residuals = self._compute_residuals(dataset, template_data, control_points, momenta)
        attachments = self._compute_individual_attachments(residuals)

        # Compute gradients if required --------------------------------------------------------------------------------
        if with_grad:
            attachment = torch.sum(attachments)
            attachment.backward()

            gradient = {}
            # Template data.
            if not self.freeze_template:
                if self.use_sobolev_gradient:
                    gradient['template_data'] = compute_sobolev_gradient(
                        template_data.grad, self.smoothing_kernel_width, self.template).data.numpy()
                else:
                    gradient['template_data'] = template_data.grad.data.numpy()

            # Control points.
            if not self.freeze_control_points: gradient['control_points'] = control_points.grad.data.numpy()

            return attachments.data.cpu().numpy(), gradient

        else:
            return attachments.data.cpu().numpy()

    def compute_sufficient_statistics(self, dataset, population_RER, individual_RER, residuals=None):
        """
        Compute the model sufficient statistics.
        """
        if residuals is None:
            # Initialize: conversion from numpy to torch ---------------------------------------------------------------
            template_data, control_points, momenta, modulation_matrix, reference_time = \
                self._fixed_effects_to_torch_tensors(False)
            sources, onset_ages, log_accelerations = self._individual_RER_to_torch_tensors(individual_RER, False)

            # Compute residuals ----------------------------------------------------------------------------------------
            residuals = self._compute_residuals(dataset, template_data, control_points, momenta, modulation_matrix,
                                                reference_time, sources, onset_ages, log_accelerations)

        # Compute sufficient statistics --------------------------------------------------------------------------------
        sufficient_statistics = {}

        # First statistical moment of the onset ages.
        onset_ages = individual_RER['onset_age']
        sufficient_statistics['S1'] = np.mean(onset_ages)

        # Second statistical moment of the onset ages.
        sufficient_statistics['S2'] = np.sum(onset_ages ** 2)

        # Second statistical moment of the log accelerations.
        log_accelerations = individual_RER['log_accelerations']
        sufficient_statistics['S3'] = np.sum(log_accelerations ** 2)

        # Second statistical moment of the residuals.
        sufficient_statistics['S4'] = np.zeros((self.number_of_objects,))
        for i in range(len(residuals)):
            for j in range(len(residuals[i])):
                for k in range(self.number_of_objects):
                    sufficient_statistics['S4'][k] += residuals[i][j][k].data.numpy()[0]

        return sufficient_statistics

    def update_fixed_effects(self, dataset, sufficient_statistics):
        """
        Updates the fixed effects based on the sufficient statistics, maximizing the likelihood.
        """
        # Covariance of the momenta update.
        prior_scale_matrix = self.priors['covariance_momenta'].scale_matrix
        prior_dof = self.priors['covariance_momenta'].degrees_of_freedom
        self.set_covariance_momenta(sufficient_statistics['S1'] + prior_dof * np.transpose(prior_scale_matrix)
                                    / (dataset.number_of_subjects + prior_dof))

        # Variance of the residual noise update.
        noise_variance = np.zeros((self.number_of_objects,))
        prior_scale_scalars = self.priors['noise_variance'].scale_scalars
        prior_dofs = self.priors['noise_variance'].degrees_of_freedom
        for k in range(self.number_of_objects):
            noise_variance[k] = (sufficient_statistics['S2'] + prior_scale_scalars[k] * prior_dofs[k]) \
                                / (dataset.number_of_subjects * self.objects_noise_dimension[k] + prior_dofs[k])
        self.set_noise_variance(noise_variance)

    def write(self, dataset, population_RER, individual_RER):
        # We save the template, the cp, the mom and the trajectories.
        sufficient_statistics = self.compute_sufficient_statistics(dataset, population_RER, individual_RER)
        self.update_fixed_effects(dataset, sufficient_statistics)
        self._write_fixed_effects(individual_RER)
        self._write_template_to_subjects_trajectories(dataset, individual_RER)  # TODO: avoid re-deforming.

    def initialize_template_attributes(self, template_specifications):
        """
        Sets the Template, TemplateObjectsName, TemplateObjectsNameExtension, TemplateObjectsNorm,
        TemplateObjectsNormKernelType and TemplateObjectsNormKernelWidth attributes.
        """

        t_list, t_name, t_name_extension, t_noise_variance, t_multi_object_attachment = \
            create_template_metadata(template_specifications)

        self.template.object_list = t_list
        self.objects_name = t_name
        self.objects_name_extension = t_name_extension
        self.multi_object_attachment = t_multi_object_attachment

        self.template.update()
        self.objects_noise_dimension = compute_noise_dimension(self.template, self.multi_object_attachment)

    ####################################################################################################################
    ### Private key methods:
    ####################################################################################################################

    def _compute_attachment(self, residuals):
        """
        Fully torch.
        """
        return torch.sum(self._compute_individual_attachments(residuals))

    def _compute_individual_attachments(self, residuals):
        """
        Fully torch.
        """
        number_of_subjects = len(residuals)
        attachments = Variable(torch.zeros((number_of_subjects,)).type(Settings().tensor_scalar_type),
                               requires_grad=False)
        for i in range(number_of_subjects):
            for j in range(len(residuals[i])):
                attachments[i] -= 0.5 * torch.sum(residuals[i][j] / Variable(
                    torch.from_numpy(self.fixed_effects['noise_variance']).type(Settings().tensor_scalar_type),
                    requires_grad=False))
        assert False  # careful check of the gradient necessary here
        return attachments

    def _compute_regularity(self, momenta):
        """
        Fully torch.
        """
        number_of_subjects = momenta.shape[0]
        regularity = 0.0

        # Momenta random effect.
        for i in range(number_of_subjects):
            regularity += self.individual_random_effects['momenta'].compute_log_likelihood_torch(momenta[i])

        # Covariance momenta prior.
        regularity += self.priors['covariance_momenta'].compute_log_likelihood(
            self.fixed_effects['covariance_momenta_inverse'])

        # Noise random effect.
        for k in range(self.number_of_objects):
            regularity -= 0.5 * self.objects_noise_dimension[k] * number_of_subjects \
                          * math.log(self.fixed_effects['noise_variance'][k])

        # Noise variance prior.
        regularity += self.priors['noise_variance'].compute_log_likelihood(self.fixed_effects['noise_variance'])

        return regularity

    def _compute_residuals(self, dataset, template_data, control_points, momenta, modulation_matrix, reference_time,
                           sources, onset_ages, log_accelerations):
        """
        Core part of the ComputeLogLikelihood methods. Fully torch.
        """

        # Initialize: longitudinal dataset -----------------------------------------------------------------------------
        targets = dataset.deformable_objects
        absolute_times = self._compute_absolute_times(dataset.times, reference_time, onset_ages, log_accelerations)

        # Deform -------------------------------------------------------------------------------------------------------
        residuals = []  # List of list of torch 1D tensors. Individuals, time-points, object.

        self.spatiotemporal_reference_frame.set_template_data_t0(template_data)
        self.spatiotemporal_reference_frame.set_control_points_t0(control_points)
        self.spatiotemporal_reference_frame.set_momenta_t0(momenta)
        self.spatiotemporal_reference_frame.set_modulation_matrix_t0(modulation_matrix)
        self.spatiotemporal_reference_frame.set_t0(reference_time)
        self.spatiotemporal_reference_frame.set_tmin(min([subject_times[0] for subject_times in absolute_times]))
        self.spatiotemporal_reference_frame.set_tmax(max([subject_times[-1] for subject_times in absolute_times]))
        self.spatiotemporal_reference_frame.update()

        for i in range(len(targets)):
            residuals.append([])
            for j, (time, target) in enumerate(zip(absolute_times[i], targets[i])):
                deformed_points = self.diffeomorphism.get_template_data(time, sources[i])
                residuals[i].append(
                    self.multi_object_attachment.compute_distances(deformed_points, self.template, target))

        return residuals

    def _compute_absolute_times(self, times, reference_time, onset_ages, log_accelerations):
        absolute_times = []
        for i in range(len(times)):
            acceleration = math.exp(log_accelerations[i])
            absolute_times.append([])
            for j in range(len(times[i])):
                absolute_times.append(acceleration * (times[i][j] - onset_ages[i]) + reference_time)
        return absolute_times

    ####################################################################################################################
    ### Private initializing methods:
    ####################################################################################################################

    def _initialize_control_points(self):
        """
        Initialize the control points fixed effect.
        """
        control_points = create_regular_grid_of_points(self.bounding_box, self.initial_cp_spacing)
        self.set_control_points(control_points)
        self.number_of_control_points = control_points.shape[0]
        print('>> Set of ' + str(self.number_of_control_points) + ' control points defined.')

    def _initialize_momenta(self):
        """
        Initialize the momenta fixed effect.
        """
        self.individual_random_effects['momenta'].mean = np.zeros(
            (self.number_of_control_points * Settings().dimension,))
        self._initialize_covariance()  # Initialize the prior and the momenta random effect.

    def _initialize_covariance(self):
        """
        Initialize the scale matrix of the inverse wishart prior, as well as the covariance matrix of the normal
        random effect.
        """
        assert self.diffeomorphism.kernel.kernel_width is not None
        dimension = Settings().dimension  # Shorthand.
        rkhs_matrix = np.zeros((self.number_of_control_points * dimension, self.number_of_control_points * dimension))
        for i in range(self.number_of_control_points):
            for j in range(self.number_of_control_points):
                cp_i = self.fixed_effects['control_points'][i, :]
                cp_j = self.fixed_effects['control_points'][j, :]
                kernel_distance = math.exp(
                    - np.sum((cp_j - cp_i) ** 2) / (self.diffeomorphism.kernel.kernel_width ** 2))  # Gaussian kernel.
                for d in range(dimension):
                    rkhs_matrix[dimension * i + d, dimension * j + d] = kernel_distance
                    rkhs_matrix[dimension * j + d, dimension * i + d] = kernel_distance
        self.priors['covariance_momenta'].scale_matrix = np.linalg.inv(rkhs_matrix)
        self.set_covariance_momenta_inverse(rkhs_matrix)

    def _initialize_noise_variance(self):
        self.set_noise_variance(np.asarray(self.priors['noise_variance'].scale_scalars))

    def _initialize_bounding_box(self):
        """
        Initialize the bounding box. which tightly encloses all template objects and the atlas control points.
        Relevant when the control points are given by the user.
        """
        assert (self.number_of_control_points > 0)
        dimension = Settings().dimension
        control_points = self.get_control_points()
        for k in range(self.number_of_control_points):
            for d in range(dimension):
                if control_points[k, d] < self.bounding_box[d, 0]:
                    self.bounding_box[d, 0] = control_points[k, d]
                elif control_points[k, d] > self.bounding_box[d, 1]:
                    self.bounding_box[d, 1] = control_points[k, d]

    ####################################################################################################################
    ### Private utility methods:
    ####################################################################################################################

    def _fixed_effects_to_torch_tensors(self, with_grad):
        """
        Convert the input fixed_effects into torch tensors.
        """
        # Template data.
        template_data = self.fixed_effects['template_data']
        template_data = Variable(torch.from_numpy(template_data).type(Settings().tensor_scalar_type),
                                 requires_grad=((not self.freeze_template) and with_grad))
        # Control points.
        control_points = self.fixed_effects['control_points']
        control_points = Variable(torch.from_numpy(control_points).type(Settings().tensor_scalar_type),
                                  requires_grad=((not self.freeze_control_points) and with_grad))
        # Momenta.
        momenta = self.fixed_effects['momenta']
        momenta = Variable(torch.from_numpy(momenta).type(Settings().tensor_scalar_type), requires_grad=with_grad)
        # Modulation matrix.
        modulation_matrix = self.fixed_effects['modulation_matrix']
        modulation_matrix = Variable(torch.from_numpy(modulation_matrix).type(Settings().tensor_scalar_type),
                                     requires_grad=with_grad)
        # Reference time.
        reference_time = self.fixed_effects['reference_time']
        reference_time = Variable(torch.from_numpy(reference_time).type(Settings().tensor_scalar_type),
                                  requires_grad=with_grad)
        return template_data, control_points, momenta, modulation_matrix, reference_time

    def _individual_RER_to_torch_tensors(self, individual_RER, with_grad):
        """
        Convert the input individual_RER into torch tensors.
        """
        # Sources.
        sources = individual_RER['sources']
        sources = Variable(torch.from_numpy(sources).type(Settings().tensor_scalar_type), requires_grad=with_grad)
        # Onset ages.
        onset_ages = individual_RER['onset_ages']
        onset_ages = Variable(torch.from_numpy(onset_ages).type(Settings().tensor_scalar_type),
                              requires_grad=with_grad)
        # Log accelerations.
        log_accelerations = individual_RER['log_accelerations']
        log_accelerations = Variable(torch.from_numpy(log_accelerations).type(Settings().tensor_scalar_type),
                                     requires_grad=with_grad)
        return sources, onset_ages, log_accelerations

    ####################################################################################################################
    ### Private writing methods:
    ####################################################################################################################

    def _write_fixed_effects(self, individual_RER):
        # Template.
        template_names = []
        for i in range(len(self.objects_name)):
            aux = self.name + "__" + self.objects_name[i] + self.objects_name_extension[i]
            template_names.append(aux)
        self.template.write(template_names)

        # Control points.
        write_2D_array(self.get_control_points(), self.name + "__control_points.txt")

        # Momenta.
        write_momenta(individual_RER['momenta'], self.name + "__momenta.txt")

        # Momenta covariance.
        write_2D_array(self.get_covariance_momenta_inverse(), self.name + "__covariance_momenta_inverse.txt")

        # Noise variance.
        write_2D_array(self.get_noise_variance(), self.name + "__noise_variance.txt")

    def _write_template_to_subjects_trajectories(self, dataset, individual_RER):
        self.diffeomorphism.set_initial_template_data_from_numpy(self.get_template_data())
        self.diffeomorphism.set_initial_control_points_from_numpy(self.get_control_points())
        for i, subject in enumerate(dataset.deformable_objects):
            names = [elt + "_to_subject_" + str(i) for elt in self.objects_name]
            self.diffeomorphism.set_initial_momenta_from_numpy(individual_RER['momenta'][i])
            self.diffeomorphism.update()
            self.diffeomorphism.write_flow(names, self.objects_name_extension, self.template)
