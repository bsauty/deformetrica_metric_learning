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
        self.fixed_effects['covariance_momenta_inverse'] = None
        self.fixed_effects['noise_variance'] = None

        # Dictionary of numpy arrays as well.
        self.priors['covariance_momenta'] = InverseWishartDistribution()
        self.priors['noise_variance'] = MultiScalarInverseWishartDistribution()

        # Dictionary of probability distributions.
        self.individual_random_effects['momenta'] = NormalDistribution()

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

    # Covariance momenta inverse ---------------------------------------------------------------------------------------
    def get_covariance_momenta_inverse(self):
        return self.fixed_effects['covariance_momenta_inverse']

    def set_covariance_momenta_inverse(self, cmi):
        self.fixed_effects['covariance_momenta_inverse'] = cmi
        self.individual_random_effects['momenta'].set_covariance_inverse(cmi)

    def set_covariance_momenta(self, cm):
        self.set_covariance_momenta_inverse(np.linalg.inv(cm))

    # Noise variance ---------------------------------------------------------------------------------------------------
    def get_noise_variance(self):
        return self.fixed_effects['noise_variance']

    def set_noise_variance(self, nv):
        self.fixed_effects['noise_variance'] = nv

    # Full fixed effects -----------------------------------------------------------------------------------------------
    def get_fixed_effects(self):
        out = {}
        if not self.freeze_template: out['template_data'] = self.fixed_effects['template_data']
        if not self.freeze_control_points: out['control_points'] = self.fixed_effects['control_points']
        return out

    def set_fixed_effects(self, fixed_effects):
        if not self.freeze_template: self.set_template_data(fixed_effects['template_data'])
        if not self.freeze_control_points: self.set_control_points(fixed_effects['control_points'])

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
    def compute_log_likelihood(self, dataset, fixed_effects, population_RER, individual_RER, with_grad=False):
        """
        Compute the log-likelihood of the dataset, given parameters fixed_effects and random effects realizations
        population_RER and indRER.

        :param dataset: LongitudinalDataset instance
        :param fixed_effects: Dictionary of fixed effects.
        :param population_RER: Dictionary of population random effects realizations.
        :param individual_RER: Dictionary of individual random effects realizations.
        :param with_grad: Flag that indicates wether the gradient should be returned as well.
        :return:
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
        momenta = Variable(torch.from_numpy(momenta).type(Settings().tensor_scalar_type), requires_grad=with_grad)

        # Deform -------------------------------------------------------------------------------------------------------
        attachment, regularity = self._compute_attachment_and_regularity(dataset, template_data, control_points,
                                                                         momenta)

        # Compute gradient if needed -----------------------------------------------------------------------------------
        if with_grad:
            total = regularity + attachment
            total.backward()

            gradient = {}
            # Template data.
            if not self.freeze_template:
                if self.use_sobolev_gradient:
                    gradient['template_data'] = compute_sobolev_gradient(
                        template_data.grad, self.smoothing_kernel_width, self.template).data.numpy()
                else:
                    gradient['template_data'] = template_data.grad.data.numpy()

            # Control points and momenta.
            if not self.freeze_control_points: gradient['control_points'] = control_points.grad.data.numpy()
            gradient['momenta'] = momenta.grad.data.cpu().numpy()

            return attachment.data.cpu().numpy()[0], regularity.data.cpu().numpy()[0], gradient

        else:
            return attachment.data.cpu().numpy()[0], regularity.data.cpu().numpy()[0]

    def compute_log_likelihood_full_torch(self, dataset, fixed_effects, population_RER, indRER):
        """
        Compute the functional. Fully torch function.
        """

        # Initialize ---------------------------------------------------------------------------------------------------
        # Template data.
        if self.freeze_template:
            template_data = Variable(torch.from_numpy(
                self.fixed_effects['template_data']).type(Settings().tensor_scalar_type), requires_grad=False)
        else:
            template_data = fixed_effects['template_data']

        # Control points.
        if self.freeze_control_points:
            control_points = Variable(torch.from_numpy(
                self.fixed_effects['control_points']).type(Settings().tensor_scalar_type), requires_grad=False)
        else:
            control_points = fixed_effects['control_points']

        # Momenta.
        momenta = fixed_effects['momenta']

        # Output -------------------------------------------------------------------------------------------------------
        return self._compute_attachment_and_regularity(dataset, template_data, control_points, momenta)

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

        attachments = Variable(torch.zeros((dataset.number_of_subjects,)).type(Settings().tensor_scalar_type),
                               requires_grad=False)
        for i in range(dataset.number_of_subjects):
            for k in range(self.number_of_objects):
                attachments[i] = attachments[i] - 0.5 * residuals[i, k] / self.fixed_effects['noise_variance'][k]

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

    def compute_sufficient_statistics(self, dataset, population_RER, individual_RER):
        """
        Compute the model sufficient statistics.
        """

        # Initialize: conversion from numpy to torch -------------------------------------------------------------------
        # Template data.
        template_data = self.fixed_effects['template_data']
        template_data = Variable(torch.from_numpy(template_data).type(Settings().tensor_scalar_type),
                                 requires_grad=False)
        # Control points.
        control_points = self.fixed_effects['control_points']
        control_points = Variable(torch.from_numpy(control_points).type(Settings().tensor_scalar_type),
                                  requires_grad=False)
        # Momenta.
        momenta = individual_RER['momenta']
        momenta = Variable(torch.from_numpy(momenta).type(Settings().tensor_scalar_type), requires_grad=False)

        # Compute residuals --------------------------------------------------------------------------------------------
        residuals = torch.sum(self._compute_residuals(dataset, template_data, control_points, momenta), dim=1)

        # Compute sufficient statistics --------------------------------------------------------------------------------
        sufficient_statistics = {}

        # Empirical momenta covariance.
        momenta = momenta.data.numpy()
        sufficient_statistics['S1'] = np.zeros((momenta[0].size, momenta[0].size))
        for i in range(dataset.number_of_subjects):
            sufficient_statistics['S1'] += np.dot(momenta[i].reshape(-1, 1), momenta[i].reshape(-1, 1).transpose())

        # Empirical residuals variances, for each object.
        residuals = residuals.data.numpy()
        sufficient_statistics['S2'] = np.zeros((self.number_of_objects,))
        for k in range(self.number_of_objects):
            sufficient_statistics['S2'][k] = residuals[k]

        # Finalization -------------------------------------------------------------------------------------------------
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

    def write(self, dataset, population_RER=None, individual_RER=None):
        # We save the template, the cp, the mom and the trajectories
        self._write_fixed_effects(individual_RER)
        self._write_template_to_subjects_trajectories(dataset, individual_RER)

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
    ### Private methods:
    ####################################################################################################################

    def _compute_attachment_and_regularity(self, dataset, template_data, control_points, momenta):
        """
        Fully torch.
        See "A Bayesian Framework for Joint Morphometry of Surface and Curve meshes in Multi-Object Complexes",
        Gori et al. (2016).
        """
        # Deform -------------------------------------------------------------------------------------------------------
        residuals = torch.sum(self._compute_residuals(dataset, template_data, control_points, momenta), dim=1)

        # Update the fixed effects for which there is a closed-form solution -------------------------------------------
        self._update_covariance_momenta(momenta.data.numpy())
        self._update_noise_variance(dataset, residuals.data.numpy())

        # Attachment part ----------------------------------------------------------------------------------------------
        attachment = 0.0
        for k in range(self.number_of_objects):
            attachment -= 0.5 * residuals[k] / self.fixed_effects['noise_variance'][k]

        # Regularity part ----------------------------------------------------------------------------------------------
        regularity = 0.0

        # Momenta random effect.
        for i in range(dataset.number_of_subjects):
            regularity += self.individual_random_effects['momenta'].compute_log_likelihood_torch(momenta[i])

        # Covariance momenta prior.
        regularity += self.priors['covariance_momenta'].compute_log_likelihood(
            self.fixed_effects['covariance_momenta_inverse'])

        # Noise random effect.
        for k in range(self.number_of_objects):
            regularity -= 0.5 * self.objects_noise_dimension[k] * dataset.number_of_subjects \
                          * math.log(self.fixed_effects['noise_variance'][k])

        # Noise variance prior.
        regularity += self.priors['noise_variance'].compute_log_likelihood(self.fixed_effects['noise_variance'])

        return attachment, regularity

    def _compute_residuals(self, dataset, template_data, control_points, momenta):
        """
        Core part of the ComputeLogLikelihood methods. Fully torch.
        """

        # Initialize: cross-sectional dataset --------------------------------------------------------------------------
        targets = dataset.deformable_objects
        targets = [target[0] for target in targets]

        # Deform -------------------------------------------------------------------------------------------------------
        residuals = Variable(torch.zeros((dataset.number_of_subjects, self.number_of_objects))
                             .type(Settings().tensor_scalar_type), requires_grad=False)

        self.diffeomorphism.set_initial_template_data(template_data)
        self.diffeomorphism.set_initial_control_points(control_points)
        for i, target in enumerate(targets):
            self.diffeomorphism.set_initial_momenta(momenta[i])
            self.diffeomorphism.update()
            deformed_points = self.diffeomorphism.get_template_data()
            residuals[i] = self.multi_object_attachment.compute_distances(deformed_points, self.template, target)

        return residuals

    def _update_covariance_momenta(self, momenta):
        """
        Fully numpy.
        """
        covariance_momenta = self.priors['covariance_momenta'].degrees_of_freedom \
                             * np.transpose(self.priors['covariance_momenta'].scale_matrix)
        for i in range(momenta.shape[0]):
            covariance_momenta += np.dot(momenta[i].reshape(-1, 1), momenta[i].reshape(-1, 1).transpose())
        covariance_momenta /= self.priors['covariance_momenta'].degrees_of_freedom + momenta.shape[0]
        self.set_covariance_momenta(covariance_momenta)

    def _update_noise_variance(self, dataset, residuals):
        """
        Fully numpy.
        """
        noise_variance = np.zeros((self.number_of_objects,))
        for k in range(self.number_of_objects):
            noise_variance[k] += self.priors['noise_variance'].degrees_of_freedom[k] \
                                 * self.priors['noise_variance'].scale_scalars[k]
            noise_variance[k] += residuals[k]
            noise_variance[k] /= self.priors['noise_variance'].degrees_of_freedom[k] \
                                 + dataset.number_of_subjects * self.objects_noise_dimension[k]
        self.set_noise_variance(noise_variance)

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

    # Write auxiliary methods ------------------------------------------------------------------------------------------
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
