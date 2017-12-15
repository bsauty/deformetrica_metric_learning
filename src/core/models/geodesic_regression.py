import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../')

import numpy as np
import math

import torch
from torch.autograd import Variable

from pydeformetrica.src.core.models.abstract_statistical_model import AbstractStatisticalModel
from pydeformetrica.src.in_out.deformable_object_reader import DeformableObjectReader
from pydeformetrica.src.in_out.dataset_functions import create_template_metadata
from pydeformetrica.src.core.model_tools.deformations.geodesic import Geodesic
from pydeformetrica.src.core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from pydeformetrica.src.support.utilities.general_settings import Settings
from pydeformetrica.src.support.utilities.initializing_functions import create_regular_grid_of_points
from pydeformetrica.src.support.kernels.kernel_functions import create_kernel
from pydeformetrica.src.in_out.utils import *
from pydeformetrica.src.core.model_tools.attachments.multi_object_attachment import MultiObjectAttachment


class GeodesicRegression(AbstractStatisticalModel):
    """
    Geodesic regression object class.
    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self):
        AbstractStatisticalModel.__init__(self)

        self.template = DeformableMultiObject()
        self.objects_name = []
        self.objects_name_extension = []
        self.objects_noise_variance = []

        self.multi_object_attachment = MultiObjectAttachment()
        self.diffeomorphism = Geodesic()

        self.smoothing_kernel_width = None
        self.initial_cp_spacing = None
        self.number_of_subjects = None
        self.number_of_objects = None
        self.number_of_control_points = None
        self.bounding_box = None

        # Dictionary of numpy arrays.
        self.fixed_effects['template_data'] = None
        self.fixed_effects['control_points'] = None
        self.fixed_effects['Momenta'] = None

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

    # Momenta ----------------------------------------------------------------------------------------------------------
    def get_momenta(self):
        return self.fixed_effects['Momenta']

    def set_momenta(self, mom):
        self.fixed_effects['Momenta'] = mom

    # Full fixed effects -----------------------------------------------------------------------------------------------
    def get_fixed_effects(self):
        out = {}
        if not (self.freeze_template):
            out['template_data'] = self.fixed_effects['template_data']
        if not (self.freeze_control_points):
            out['control_points'] = self.fixed_effects['control_points']
        out['Momenta'] = self.fixed_effects['Momenta']
        return out

    def set_fixed_effects(self, fixed_effects):
        if not (self.freeze_template):
            self.set_template_data(fixed_effects['template_data'])
        if not (self.freeze_control_points):
            self.set_control_points(fixed_effects['control_points'])
        self.set_momenta(fixed_effects['Momenta'])

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def update(self):
        """
        Final initialization steps.
        """

        self.template.update()
        self.number_of_objects = len(self.template.object_list)
        self.bounding_box = self.template.bounding_box

        self.set_template_data(self.template.get_data())

        if self.fixed_effects['control_points'] is None: self._initialize_control_points()
        else: self._initialize_bounding_box()

        if self.fixed_effects['Momenta'] is None: self._initialize_momenta()

    # Compute the functional. Numpy input/outputs.
    def compute_log_likelihood(self, dataset, fixed_effects, pop_RER=None, ind_RER=None, with_grad=False):
        """
        Compute the log-likelihood of the dataset, given parameters fixed_effects and random effects realizations
        pop_RER and indRER.

        :param dataset: LongitudinalDataset instance
        :param fixed_effects: Dictionary of fixed effects.
        :param pop_RER: Dictionary of population random effects realizations.
        :param indRER: Dictionary of individual random effects realizations.
        :param with_grad: Flag that indicates wether the gradient should be returned as well.
        :return:
        """

        # Initialize: conversion from numpy to torch -------------------------------------------------------------------
        # Template data.
        if not (self.freeze_template):
            template_data = fixed_effects['template_data']
            template_data = Variable(torch.from_numpy(template_data).type(Settings().tensor_scalar_type),
                                     requires_grad=with_grad)
        else:
            template_data = self.fixed_effects['template_data']
            template_data = Variable(torch.from_numpy(template_data).type(Settings().tensor_scalar_type),
                                     requires_grad=False)

        # Control points.
        if not (self.freeze_control_points):
            control_points = fixed_effects['control_points']
            control_points = Variable(torch.from_numpy(control_points).type(Settings().tensor_scalar_type),
                                      requires_grad=with_grad)
        else:
            control_points = self.fixed_effects['control_points']
            control_points = Variable(torch.from_numpy(control_points).type(Settings().tensor_scalar_type),
                                      requires_grad=False)

        # Momenta.
        momenta = fixed_effects['Momenta']
        momenta = Variable(torch.from_numpy(momenta).type(Settings().tensor_scalar_type), requires_grad=with_grad)

        # Deform -------------------------------------------------------------------------------------------------------
        regularity, attachment = self._compute_attachement_and_regularity(dataset, template_data, control_points,
                                                                          momenta)

        # Compute gradient if needed -----------------------------------------------------------------------------------
        if with_grad:
            total = regularity + attachment
            total.backward()

            gradient = {}
            if not (self.freeze_template): gradient['template_data'] = template_data.grad.data.numpy()
            if not (self.freeze_control_points): gradient['control_points'] = control_points.grad.data.numpy()
            gradient['Momenta'] = momenta.grad.data.cpu().numpy()

            return attachment.data.cpu().numpy()[0], regularity.data.cpu().numpy()[0], gradient

        else:
            return attachment.data.cpu().numpy()[0], regularity.data.cpu().numpy()[0]

    def compute_log_likelihood_full_torch(self, dataset, fixed_effects, pop_RER, indRER):
        """
        Compute the functional. Fully torch function.
        """

        # Initialize ---------------------------------------------------------------------------------------------------
        # Template data.
        if self.freeze_template:
            template_data = Variable(torch.from_numpy(self.fixed_effects['template_data']), requires_grad=False)
        else:
            template_data = fixed_effects['template_data']

        # Control points.
        if self.freeze_control_points:
            control_points = Variable(torch.from_numpy(self.fixed_effects['control_points']), requires_grad=False)
        else:
            control_points = fixed_effects['control_points']

        # Momenta.
        momenta = fixed_effects['Momenta']

        # Output -------------------------------------------------------------------------------------------------------
        return self._compute_attachement_and_regularity(dataset, template_data, control_points, momenta)

    # def convolve_grad_template(gradTemplate):
    #     """
    #     Smoothing of the template gradient (for landmarks)
    #     """
    #     grad_template_sob = []
    #
    #     kernel = TorchKernel()
    #     kernel.KernelWidth = self.SmoothingKernelWidth
    #     template_data = self.get_template_data()
    #     pos = 0
    #     for elt in tempData:
    #         # TODO : assert if data is image or not.
    #         grad_template_sob.append(kernel.convolve(
    #             template_data, template_data, gradTemplate[pos:pos + len(template_data)]))
    #         pos += len(template_data)
    #     return gradTemplate

    def write(self, dataset):
        # We save the template, the cp, the mom and the trajectories
        self._write_template()
        self._write_control_points()
        self._write_momenta()
        self._write_geodesic_flow(dataset)

    def initialize_template_attributes(self, template_specifications):
        """
        Sets the Template, TemplateObjectsName, TemplateObjectsNameExtension, TemplateObjectsNorm,
        TemplateObjectsNormKernelType and TemplateObjectsNormKernelWidth attributes.
        """

        t_list, t_name, t_name_extension, t_noise_variance, t_norm, t_multi_object_attachment = \
            create_template_metadata(template_specifications)

        self.template.object_list = t_list
        self.objects_name = t_name
        self.objects_name_extension = t_name_extension
        self.objects_noise_variance = t_noise_variance
        self.multi_object_attachment = t_multi_object_attachment

    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    def _compute_attachement_and_regularity(self, dataset, template_data, control_points, momenta):
        """
        Core part of the ComputeLogLikelihood methods. Fully torch.
        """

        # Initialize: cross-sectional dataset --------------------------------------------------------------------------
        target_times = dataset.times[0]
        target_objects = dataset.deformable_objects[0]

        # Deform -------------------------------------------------------------------------------------------------------
        self.diffeomorphism.tmin = min(target_times)
        self.diffeomorphism.tmax = max(target_times)
        self.diffeomorphism.template_data_t0 = template_data
        self.diffeomorphism.control_points_t0 = control_points
        self.diffeomorphism.momenta_t0 = momenta
        self.diffeomorphism.update()

        attachment = 0.
        for j, (time, object) in enumerate(zip(target_times, target_objects)):
            deformedPoints = self.diffeomorphism.get_template_data(time)
            attachment -= self.multi_object_attachment.compute_weighted_distance(
                deformedPoints, self.template, object, self.objects_noise_variance)

        regularity = - self.diffeomorphism.get_norm()

        return attachment, regularity

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
        assert (self.number_of_subjects > 0)
        momenta = np.zeros((self.number_of_control_points, Settings().dimension))
        self.set_momenta(momenta)

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
    def _write_template(self):
        templateNames = []
        for i in range(len(self.objects_name)):
            aux = self.name + '_' + self.objects_name[i] + self.objects_name_extension[i]
            templateNames.append(aux)
        self.template.write(templateNames)

    def _write_control_points(self):
        write_2D_array(self.get_control_points(), self.name + "__control_points.txt")

    def _write_momenta(self):
        write_2D_array(self.get_momenta(), self.name + "__momenta.txt")

    def _write_geodesic_flow(self, dataset):
        template_data = Variable(torch.from_numpy(self.get_template_data()), requires_grad=False)
        control_points = Variable(torch.from_numpy(self.get_control_points()), requires_grad=False)
        momenta = Variable(torch.from_numpy(self.get_momenta()), requires_grad=False)

        target_times = dataset.times[0]
        self.diffeomorphism.tmin = min(target_times)
        self.diffeomorphism.tmax = max(target_times)
        self.diffeomorphism.template_data_t0 = template_data
        self.diffeomorphism.control_points_t0 = control_points
        self.diffeomorphism.momenta_t0 = momenta
        self.diffeomorphism.update()
        self.diffeomorphism.write_flow(self.name, self.objects_name, self.objects_name_extension, self.template)
