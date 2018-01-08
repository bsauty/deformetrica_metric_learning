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
from pydeformetrica.src.core.model_tools.deformations.exponential import Exponential
from pydeformetrica.src.core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from pydeformetrica.src.support.utilities.general_settings import Settings
from pydeformetrica.src.core.models.model_functions import create_regular_grid_of_points, compute_sobolev_gradient
from pydeformetrica.src.support.kernels.kernel_functions import create_kernel
from pydeformetrica.src.in_out.utils import *
from pydeformetrica.src.core.model_tools.attachments.multi_object_attachment import MultiObjectAttachment
from copy import deepcopy
from torch.multiprocessing import Process, SimpleQueue, Queue, Pool, Manager
import time


def _subject_attachment_and_regularity(arg):
    """
    auxiliary function for multiprocessing (cannot be a class method)
    """
    (i, template, template_data, mom, cps, target, multi_object_attachment, objects_noise_variance, diffeo, q, with_grad) = arg
    # start_time = time.time()
    diffeo.set_initial_template_data(template_data)
    diffeo.set_initial_control_points(cps)
    diffeo.set_initial_momenta(mom)
    diffeo.update()
    deformed_points = diffeo.get_template_data()
    attachment = -1. * multi_object_attachment.compute_weighted_distance(
        deformed_points, template, target, objects_noise_variance)
    regularity = -1. * diffeo.get_norm_squared()
    total_for_subject = attachment + regularity
    grad_mom = grad_cps = grad_template_data = None
    if with_grad:
        # Computing the gradient
        total_for_subject.backward()
        # Those gradients are none if requires_grad=False
        grad_template_data = template_data.grad
        grad_cps = cps.grad
        grad_mom = mom.grad
    q.put((i, attachment, regularity, grad_mom, grad_cps, grad_template_data))
    # end_time = time.time()
    # print("Process", i, "took", end_time - start_time,  "seconds", start_time, end_time)


class DeterministicAtlas(AbstractStatisticalModel):
    """
    Deterministic atlas object class.

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
        self.diffeomorphism = Exponential()

        self.use_sobolev_gradient = True
        self.smoothing_kernel_width = None

        self.initial_cp_spacing = None
        self.number_of_subjects = None
        self.number_of_objects = None
        self.number_of_control_points = None
        self.bounding_box = None

        # Dictionary of numpy arrays.
        self.fixed_effects['template_data'] = None
        self.fixed_effects['control_points'] = None
        self.fixed_effects['momenta'] = None

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

    # Full fixed effects -----------------------------------------------------------------------------------------------
    def get_fixed_effects(self):
        out = {}
        if not (self.freeze_template):
            out['template_data'] = self.fixed_effects['template_data']
        if not (self.freeze_control_points):
            out['control_points'] = self.fixed_effects['control_points']
        out['momenta'] = self.fixed_effects['momenta']
        return out

    def set_fixed_effects(self, fixed_effects):
        if not (self.freeze_template):
            self.set_template_data(fixed_effects['template_data'])
        if not (self.freeze_control_points):
            self.set_control_points(fixed_effects['control_points'])
        self.set_momenta(fixed_effects['momenta'])

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def update(self):
        self.steps__ = """
        Final initialization steps.
        """

        self.template.update()
        self.number_of_objects = len(self.template.object_list)
        self.bounding_box = self.template.bounding_box

        self.set_template_data(self.template.get_points())
        if self.fixed_effects['control_points'] is None:
            self._initialize_control_points()
        else:
            self._initialize_bounding_box()
        if self.fixed_effects['momenta'] is None: self._initialize_momenta()

    # Compute the functional. Numpy input/outputs.
    def compute_log_likelihood(self, dataset, fixed_effects, population_RER=None, individual_RER=None, with_grad=False):
        """
        Compute the log-likelihood of the dataset, given parameters fixed_effects and random effects realizations
        population_RER and indRER.

        :param dataset: LongitudinalDataset instance
        :param fixed_effects: Dictionary of fixed effects.
        :param population_RER: Dictionary of population random effects realizations.
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
        momenta = fixed_effects['momenta']
        momenta = Variable(torch.from_numpy(momenta).type(Settings().tensor_scalar_type), requires_grad=with_grad)

        # Deform -------------------------------------------------------------------------------------------------------
        if with_grad:
            attachment, regularity, gradient = self._compute_attachment_and_regularity(
                dataset, template_data, control_points, momenta, with_grad=True)
            return attachment, regularity, gradient

        else:
            attachment, regularity, _ = self._compute_attachment_and_regularity(
                dataset, template_data, control_points, momenta, with_grad=False)

            return attachment, regularity

    def convolve_grad_template(self, grad_template):
        """
        Smoothing of the template gradient (for landmarks with boundaries)
        """
        grad_template_sob = []

        aux = self.diffeomorphism.kernel.kernel_width
        self.diffeomorphism.kernel.kernel_width = self.smoothing_kernel_width
        template_data = Variable(torch.from_numpy(self.get_template_data()).type(Settings().tensor_scalar_type))
        pos = 0
        for elt in self.template.object_list:
            grad_template_sob.append(self.diffeomorphism.kernel.convolve(
                template_data[pos:pos + len(elt.get_points())],
                template_data[pos:pos + len(elt.get_points())],
                grad_template[pos:pos + len(elt.get_points())]))
            pos += len(elt.get_points())
        self.diffeomorphism.kernel.kernel_width = aux
        return grad_template

    def write(self, dataset, population_RER=None, individual_RER=None):
        # We save the template, the cp, the mom and the trajectories
        self._write_template()
        self._write_control_points()
        self._write_momenta()
        self._write_template_to_subjects_trajectories(dataset)

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
        self.objects_noise_variance = t_noise_variance
        self.multi_object_attachment = t_multi_object_attachment
        self.template.update()

    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    def _compute_attachment_and_regularity(self, dataset, template_data, control_points, momenta, with_grad=False):
        """
        Core part of the ComputeLogLikelihood methods. Torch input, numpy output. TODO: put numpy input ! maybe factorise the two methods.
        """
        # Initialize: cross-sectional dataset --------------------------------------------------------------------------
        targets = [target[0] for target in dataset.deformable_objects]

        if Settings().number_of_threads > 1:
            torch.set_num_threads(1)# Because it's better to parallelize top level ops

        pool = Pool(processes=Settings().number_of_threads)
        m = Manager()
        # Queue used to store the results.
        q = m.Queue()

        # Copying all the arguments, maybe some deep copies are avoidable
        args = [(i, deepcopy(self.template), template_data.clone(), momenta[i].clone(), control_points.clone(),
                 targets[i], self.multi_object_attachment, self.objects_noise_variance,
                 deepcopy(self.diffeomorphism), q, with_grad) for i in range(len(targets))]

        pool.map(_subject_attachment_and_regularity, args)

        regularity = 0.
        attachment = 0.
        gradient = {}
        gradient_numpy = {}

        # Loop to gather the results
        nb_ended_workers = 0
        while nb_ended_workers != len(targets):
            worker_result = q.get()
            if worker_result is None:
                pass
            else:
                i, attachment_for_target, regularity_for_target, grad_mom, grad_cps, grad_template_data = worker_result
                nb_ended_workers += 1
                attachment += attachment_for_target
                regularity += regularity_for_target
                if with_grad:
                    if grad_mom is not None:
                        if 'momenta' not in gradient.keys():
                            gradient['momenta'] = torch.zeros_like(momenta)
                        gradient['momenta'][i] = grad_mom

                    if grad_cps is not None:
                        if 'control_points' not in gradient.keys():
                            gradient['control_points'] = torch.zeros_like(control_points)
                        gradient['control_points'] += grad_cps

                    if grad_template_data is not None:
                        if 'template_data' not in gradient.keys():
                            gradient['template_data'] = torch.zeros_like(template_data)
                        gradient['template_data'] += grad_template_data

        if with_grad:
            if not self.freeze_template and self.use_sobolev_gradient:
                gradient['template_data'] = self.convolve_grad_template(gradient['template_data'])

            for (key, value) in gradient.items():
                gradient_numpy[key] = value.data.numpy()

        return attachment.data.numpy()[0], regularity.data.numpy()[0], gradient_numpy

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
        momenta = np.zeros(
            (self.number_of_subjects, self.number_of_control_points, Settings().dimension))
        self.set_momenta(momenta)
        print('>> Deterministic atlas momenta initialized to zero, for ' + str(self.number_of_subjects) + ' subjects.')

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
        template_names = []
        for i in range(len(self.objects_name)):
            aux = self.name + "_" + self.objects_name[i] + self.objects_name_extension[i]
            template_names.append(aux)
        self.template.write(template_names)

    def _write_control_points(self):
        write_2D_array(self.get_control_points(), self.name + "__control_points.txt")

    def _write_momenta(self):
        write_momenta(self.get_momenta(), self.name + "__momenta.txt")

    def _write_template_to_subjects_trajectories(self, dataset):
        self.diffeomorphism.set_initial_control_points_from_numpy(self.get_control_points())
        self.diffeomorphism.set_initial_template_data_from_numpy(self.get_template_data())

        for i, subject in enumerate(dataset.deformable_objects):
            names = [elt + "_to_subject_" + str(i) for elt in self.objects_name]
            self.diffeomorphism.set_initial_momenta_from_numpy(self.get_momenta()[i])
            self.diffeomorphism.update()
            self.diffeomorphism.write_flow(names, self.objects_name_extension, self.template)

