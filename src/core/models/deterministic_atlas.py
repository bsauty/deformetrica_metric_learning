import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../')

import numpy as np
import math
from copy import deepcopy
import time

import torch
from torch.autograd import Variable
from torch.multiprocessing import Pool

from pydeformetrica.src.core.models.abstract_statistical_model import AbstractStatisticalModel
from pydeformetrica.src.in_out.deformable_object_reader import DeformableObjectReader
from pydeformetrica.src.in_out.dataset_functions import create_template_metadata
from pydeformetrica.src.core.model_tools.deformations.exponential import Exponential
from pydeformetrica.src.core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from pydeformetrica.src.support.utilities.general_settings import Settings
from pydeformetrica.src.core.models.model_functions import create_regular_grid_of_points, compute_sobolev_gradient
from pydeformetrica.src.support.kernels.kernel_functions import create_kernel
from pydeformetrica.src.in_out.array_readers_and_writers import *
from pydeformetrica.src.core.model_tools.attachments.multi_object_attachment import MultiObjectAttachment


def _subject_attachment_and_regularity(arg):
    """
    auxiliary function for multiprocessing (cannot be a class method)
    """
    (settings, template, template_data, template_points, mom, cps, target, multi_object_attachment,
     objects_noise_variance, diffeo, with_grad) = arg
    Settings().initialize(settings)

    # start_time = time.time()
    diffeo.set_initial_template_points(template_points)
    diffeo.set_initial_control_points(cps)
    diffeo.set_initial_momenta(mom)
    diffeo.update()

    deformed_points = diffeo.get_template_points()
    deformed_data = template.get_deformed_data(deformed_points, template_data)

    attachment = - multi_object_attachment.compute_weighted_distance(deformed_data, template, target,
                                                                     objects_noise_variance)
    regularity = - diffeo.get_norm_squared()
    total_for_subject = attachment + regularity

    grad_mom = grad_cps = grad_template_data = None
    if with_grad:
        # Computing the gradient
        total_for_subject.backward()
        # Those gradients are none if requires_grad=False
        grad_template_data = {key: value.grad for key, value in template_data.items()}
        grad_cps = cps.grad
        grad_mom = mom.grad

    return attachment, regularity, grad_mom, grad_cps, grad_template_data
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
        self.exponential = Exponential()

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
        self.freeze_momenta = False

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
        if not self.freeze_template:
            for key, value in self.fixed_effects['template_data'].items():
                out[key] = value
        if not self.freeze_control_points:
            out['control_points'] = self.fixed_effects['control_points']
        if not self.freeze_momenta:
            out['momenta'] = self.fixed_effects['momenta']
        return out

    def set_fixed_effects(self, fixed_effects):
        if not self.freeze_template:
            template_data = {key: fixed_effects[key] for key in self.fixed_effects['template_data'].keys()}
            self.set_template_data(template_data)
        if not self.freeze_control_points:
            self.set_control_points(fixed_effects['control_points'])
        if not self.freeze_momenta:
            self.set_momenta(fixed_effects['momenta'])

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
        if self.fixed_effects['control_points'] is None:
            self._initialize_control_points()
        else:
            self._initialize_bounding_box()
        if self.fixed_effects['momenta'] is None: self._initialize_momenta()

    # Compute the functional. Numpy input/outputs.
    def compute_log_likelihood(self, dataset, population_RER, individual_RER, mode='complete', with_grad=False):
        """
        Compute the log-likelihood of the dataset, given parameters fixed_effects and random effects realizations
        population_RER and indRER.

        :param dataset: LongitudinalDataset instance
        :param fixed_effects: Dictionary of fixed effects.
        :param population_RER: Dictionary of population random effects realizations.
        :param individual_RER: Dictionary of individual random effects realizations.
        :param mode: Indicates which log_likelihood should be computed, between 'complete', 'model', and 'class2'.
        :param with_grad: Flag that indicates wether the gradient should be returned as well.
        :return:
        """

        # Initialize: conversion from numpy to torch -------------------------------------------------------------------
        template_data, template_points, control_points, momenta = self._fixed_effects_to_torch_tensors(with_grad)

        # Deform -------------------------------------------------------------------------------------------------------
        if with_grad:
            attachment, regularity, gradient = self._compute_attachment_and_regularity(
                dataset, template_data, template_points, control_points, momenta, with_grad=True)
            return attachment, regularity, gradient

        else:
            attachment, regularity, _ = self._compute_attachment_and_regularity(
                dataset, template_data, template_points, control_points, momenta, with_grad=False)

            return attachment, regularity

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

    def _compute_attachment_and_regularity(self, dataset, template_data, template_points, control_points, momenta,
                                           with_grad=False):
        """
        Core part of the ComputeLogLikelihood methods. Torch input, numpy output.
        """
        # Initialize: cross-sectional dataset --------------------------------------------------------------------------
        targets = [target[0] for target in dataset.deformable_objects]

        # Output initialization
        regularity = 0.
        attachment = 0.
        gradient = {}
        gradient_numpy = {}

        # Multi-threaded version
        if Settings().number_of_threads > 1:

            # Pool of jobs.
            pool = Pool(processes=Settings().number_of_threads)

            # Copying all the arguments, maybe some deep copies are avoidable
            args = [(Settings().serialize(), self.template.clone(),
                     {key: value.clone() for key, value in template_data.items()},
                     {key: value.clone() for key, value in template_points.items()},
                     momenta[i].clone(), control_points.clone(), targets[i],
                     self.multi_object_attachment, self.objects_noise_variance,
                     self.exponential.light_copy(), with_grad) for i in range(len(targets))]

            results = pool.map(_subject_attachment_and_regularity, args)
            pool.close()
            pool.join()

            for i in range(self.number_of_subjects):
                attachment += results[i][0]
                regularity += results[i][1]

            if with_grad:
                if not self.freeze_momenta and results[0][2] is not None:
                    gradient['momenta'] = torch.zeros_like(momenta)
                    gradient['momenta'][0] = results[0][2]
                    for i in range(1, self.number_of_subjects): gradient['momenta'][i] = results[i][2]

                if not self.freeze_control_points and results[0][3] is not None:
                    gradient['control_points'] = results[0][3]
                    for i in range(1, self.number_of_subjects): gradient['control_points'] += results[i][3]

                if not self.freeze_template and results[0][4] is not None:
                    for key, value in results[0][4].items():
                        gradient[key] = value
                        for i in range(1, self.number_of_subjects): gradient[key] += results[i][4][key]

        # Single thread version (to avoid overhead in this case)
        else:
            self.exponential.set_initial_template_points(template_points)
            self.exponential.set_initial_control_points(control_points)

            for i, target in enumerate(targets):
                self.exponential.set_initial_momenta(momenta[i])
                self.exponential.update()
                deformed_points = self.exponential.get_template_points()
                deformed_data = self.template.get_deformed_data(deformed_points, template_data)
                regularity -= self.exponential.get_norm_squared()
                attachment -= self.multi_object_attachment.compute_weighted_distance(
                    deformed_data, self.template, target, self.objects_noise_variance)

            total = attachment + regularity
            if with_grad:
                total.backward()
                if not self.freeze_momenta and momenta.grad is not None:
                    gradient['momenta'] = momenta.grad
                if not self.freeze_control_points and control_points.grad is not None:
                    gradient['control_points'] = control_points.grad
                if not self.freeze_template:
                    for key, value in template_data.items():
                        if value.grad is not None:
                            gradient[key] = value.grad

        if with_grad:
            if not self.freeze_template and self.use_sobolev_gradient and 'template_landmark_points' in gradient.keys():
                gradient['template_landmark_points'] = compute_sobolev_gradient(
                    gradient['template_landmark_points'], self.smoothing_kernel_width, self.template)

            for (key, value) in gradient.items():
                gradient_numpy[key] = value.data.numpy()

        return attachment.data.numpy()[0], regularity.data.numpy()[0], gradient_numpy

    def _initialize_control_points(self):
        """
        Initialize the control points fixed effect.
        """
        if not Settings().dense_mode:
            control_points = create_regular_grid_of_points(self.bounding_box, self.initial_cp_spacing)
        else:
            control_points = self.template.get_points()

        # FILTERING TOO CLOSE POINTS: DISABLED FOR NOW

        # indices_to_remove = []
        # for i in range(len(control_points)):
        #     for j in range(len(control_points)):
        #         if i != j:
        #             d = np.linalg.norm(control_points[i] - control_points[j])
        #             if d < 0.1 * self.exponential.kernel.kernel_width:
        #                 indices_to_remove.append(i)
        #
        # print(len(indices_to_remove))
        #
        # indices_to_remove = list(set(indices_to_remove))
        # indices_to_keep = [elt for elt in range(len(control_points)) if elt not in indices_to_remove]
        # control_points = np.array([control_points[i] for i in indices_to_keep])

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

    ####################################################################################################################
    ### Private utility methods:
    ####################################################################################################################

    def _fixed_effects_to_torch_tensors(self, with_grad):
        """
        Convert the fixed_effects into torch tensors.
        """
        # Template data.
        template_data = self.fixed_effects['template_data']
        template_data = {key: Variable(torch.from_numpy(value).type(Settings().tensor_scalar_type),
                                       requires_grad=(not self.freeze_template and with_grad))
                         for key, value in template_data.items()}

        # Template points.
        template_points = self.template.get_points()
        template_points = {key: Variable(torch.from_numpy(value).type(Settings().tensor_scalar_type),
                                         requires_grad=(not self.freeze_template and with_grad))
                           for key, value in template_points.items()}

        # Control points.
        if Settings().dense_mode:
            assert 'image_intensities' not in template_data.keys() and 'image_points' not in template_points.keys(), \
                'Dense mode not available with image data.'
            control_points = template_data
        else:
            control_points = self.fixed_effects['control_points']
            control_points = Variable(torch.from_numpy(control_points).type(Settings().tensor_scalar_type),
                                      requires_grad=(not self.freeze_control_points and with_grad))
        # Momenta.
        momenta = self.fixed_effects['momenta']
        momenta = Variable(torch.from_numpy(momenta).type(Settings().tensor_scalar_type),
                           requires_grad=(not self.freeze_momenta and with_grad))

        return template_data, template_points, control_points, momenta

    ####################################################################################################################
    ### Writing methods:
    ####################################################################################################################

    def write(self, dataset, population_RER, individual_RER):
        # We save the template, the cp, the mom and the trajectories
        self._write_fixed_effects()
        self._write_template_to_subjects_trajectories(dataset)

    def _write_fixed_effects(self):
        # Template.
        template_names = []
        for i in range(len(self.objects_name)):
            aux = self.name + "__EstimatedParameters__Template_" + self.objects_name[i] + self.objects_name_extension[i]
            template_names.append(aux)
        self.template.write(template_names)

        # Control points.
        write_2D_array(self.get_control_points(), self.name + "__EstimatedParameters__ControlPoints.txt")

        # Momenta.
        write_3D_array(self.get_momenta(), self.name + "__EstimatedParameters__Momenta.txt")

        # Writing the first momenta for each subject as a vtk file for visualization purposes.
        write_control_points_and_momenta_vtk(self.get_control_points(), self.get_momenta()[0],
                                             self.name + "__EstimatedParameters__ControlPointsAndMomenta.vtk")

    def _write_template_to_subjects_trajectories(self, dataset):
        template_data, template_points, control_points, momenta = self._fixed_effects_to_torch_tensors(False)

        self.exponential.set_initial_template_points(template_points)
        self.exponential.set_initial_control_points(control_points)

        for i, subject in enumerate(dataset.deformable_objects):
            names = [self.name + '__' + elt + "_to_subject_" + str(i) for elt in self.objects_name]
            self.exponential.set_initial_momenta(momenta[i])
            self.exponential.update()
            self.exponential.write_flow(names, self.objects_name_extension, self.template, template_data)
