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


# MULTIPROCESSING, WORK IN PROGRESS
def _subject_attachment_and_regularity(arg):
    (i, template, template_data, mom, cps, target, multi_object_attachment, objects_noise_variance, diffeo, q, with_grad) = arg
    # Lists are thread-safe for reading (not if the data is modified)
    # Tensors are thread-safe
    start_time = time.time()
    diffeo.set_initial_template_data(template_data)
    diffeo.set_initial_control_points(cps)
    diffeo.set_initial_momenta(mom[i])
    diffeo.update()
    deformed_points = diffeo.get_template_data()
    attachment = -1. * multi_object_attachment.compute_weighted_distance(
        deformed_points, template, target, objects_noise_variance)
    regularity = -1. * diffeo.get_norm_squared()
    total_for_subject = attachment + regularity
    grad = {}
    if with_grad:
        if mom.grad:
            mom.grad.data.zero_()
        if template_data.grad:
            template_data.grad.data.zero_()
        if cps.grad:
            cps.grad.data.zero_()
        total_for_subject.backward()
        # we need the three grads: template_data, mom and control_points
        if with_grad:
            if template_data.requires_grad:
                grad['template_data'] = template_data.grad
            if cps.requires_grad:
                grad['control_points'] = cps.grad
            grad['momenta'] = mom.grad
    q.put([attachment, regularity, grad])
    end_time = time.time()
    print("Process", i, "took", end_time - start_time,  "seconds", start_time, end_time)


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
            attachment, regularity, gradient = self._compute_attachement_and_regularity(dataset, template_data, control_points,
                                                                          momenta, with_grad=True)
            return attachment, regularity, gradient

            # if not self.freeze_template:
            #     if self.use_sobolev_gradient:
        # # Compute gradient if needed -----------------------------------------------------------------------------------
        # if with_grad:
        #     total = regularity + attachment
        #     total.backward()
        #
        #     gradient = {}
        #     # Template data.
        #     if not self.freeze_template:
        #         if self.use_sobolev_gradient:
        #             gradient['template_data'] = compute_sobolev_gradient(
        #                 template_data.grad, self.smoothing_kernel_width, self.template).data.numpy()
        #         else:
        #             gradient['template_data'] = template_data.grad.data.numpy()
        #
        #     # Control points and momenta.
        #     if not self.freeze_control_points: gradient['control_points'] = control_points.grad.data.numpy()
        #     gradient['momenta'] = momenta.grad.data.cpu().numpy()

            # return attachment.data.cpu().numpy()[0], regularity.data.cpu().numpy()[0], gradient

        else:
            attachment, regularity, _ = self._compute_attachement_and_regularity(dataset, template_data, control_points,
                                                                          momenta, with_grad=False)

            return attachment, regularity

    def compute_log_likelihood_full_torch(self, dataset, fixed_effects, population_RER, indRER):
        """
        Compute the functional. Fully torch function.
        """
        assert False, "Torch BFGS not maintained any more."
        # # Initialize ---------------------------------------------------------------------------------------------------
        # # Template data.
        # if self.freeze_template:
        #     template_data = Variable(torch.from_numpy(
        #         self.fixed_effects['template_data']).type(Settings().tensor_scalar_type), requires_grad=False)
        # else:
        #     template_data = fixed_effects['template_data']
        #
        # # Control points.
        # if self.freeze_control_points:
        #     control_points = Variable(torch.from_numpy(
        #         self.fixed_effects['control_points']).type(Settings().tensor_scalar_type), requires_grad=False)
        # else:
        #     control_points = fixed_effects['control_points']
        #
        # # Momenta.
        # momenta = fixed_effects['momenta']
        #
        # # Output -------------------------------------------------------------------------------------------------------
        # return self._compute_attachement_and_regularity(dataset, template_data, control_points, momenta)

    def convolve_grad_template(gradTemplate):
        """
        Smoothing of the template gradient (for landmarks with boundaries)
        """
        grad_template_sob = []

        kernel = ExactKernel()
        kernel.KernelWidth = self.SmoothingKernelWidth
        template_data = self.get_template_data()
        pos = 0
        for elt in tempData:
            # TODO : assert if data is image or not.
            grad_template_sob.append(kernel.convolve(
                template_data, template_data, gradTemplate[pos:pos + len(template_data)]))
            pos += len(template_data)
        return gradTemplate

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

    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    def _compute_attachement_and_regularity(self, dataset, template_data, control_points, momenta, with_grad=False):
        """
        Core part of the ComputeLogLikelihood methods. Torch input, numpy output. TODO: put numpy input ! maybe factorise the two methods.
        """

        # Initialize: cross-sectional dataset --------------------------------------------------------------------------
        targets = dataset.deformable_objects
        targets = [target[0] for target in targets]
        nb_observations = len(targets)

        if Settings().number_of_threads > 1:
            torch.set_num_threads(1)# Because it's better to parallelize top level ops

        print("Using", Settings().number_of_threads, " Thread(s)")

        pool = Pool(processes=Settings().number_of_threads)
        m = Manager()
        q = m.Queue()


        # This massive copy could take some space !!! Maybe avoidable if the objets are thread-safe ?
        args = [(i, deepcopy(self.template), template_data.clone(), momenta.clone(), control_points.clone(), targets[i], self.multi_object_attachment,
                    self.objects_noise_variance, deepcopy(self.diffeomorphism), q, with_grad) for i in range(nb_observations)]

        pool.map(_subject_attachment_and_regularity, args)

        regularity = 0.
        attachment = 0.
        gradient = {}

        nb_ended_workers = 0
        while nb_ended_workers != len(targets):
            worker_result = q.get()
            if worker_result is None:
                pass
            else:
                nb_ended_workers += 1
                attachment += worker_result[0]
                regularity += worker_result[1]
                # If the gradient was not required, worker_result[2] is an empty dictionary.
                if not gradient:
                    # empty dictionaries evaluate to False
                    gradient = worker_result[2]
                else:
                    assert len(gradient) == len(worker_result[2])
                    for (key, value) in worker_result[2].items():
                        gradient[key] += value

        gradient_numpy = {}
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
#
#
# # Initialize: cross-sectional dataset --------------------------------------------------------------------------
# targets = dataset.deformable_objects
# targets = [target[0] for target in targets]
#
# # Queue to store the results
# q = Queue()
# # Event to keep the processes alive while we haven't fully processed the output
# event = Event()
#
# # TODO : note that for now, momenta.grad is very very sparse...
#
# # print("Process", i, "done", attachment.data.numpy()[0], regularity.data.numpy()[0])
# event.wait()
# # print("Process", i, "Done waiting")
#
#
# processes = []
# for i in range(len(momenta)):
#     process = Process(target=_subject_attachment_and_regularity, args=(i, template_data, momenta, control_points, q))
#     processes.append(process)
#     process.start()
#
# regularity = 0.
# attachment = 0.
# gradient = {}
#
# nb_ended_workers = 0
# while nb_ended_workers != len(targets):
#     worker_result = q.get()
#     if worker_result is None:
#         pass
#     else:
#         nb_ended_workers += 1
#         attachment += worker_result[0]
#         regularity += worker_result[1]
#         # If the gradient was not required, worker_result[2] is an empty dictionary.
#         if not gradient:
#             # empty dictionaries evaluate to False
#             gradient = worker_result[2]
#         else:
#             assert len(gradient) == len(worker_result[2])
#             for (key, value) in worker_result[2].items():
#                 gradient[key] += value
#
# # Now that we gathered all the results, we trigger the event so that the processes can finish.
# event.set()
#
# # This is the proper way to make sure all processes have finished.
# for process in processes:
#     process.join()
#
# # At this point, we have attachment, regularity and gradient in torch types.
# # Before switching to numpy, we convolve the template gradient !
# if not self.freeze_template:
#     if self.use_sobolev_gradient:
#         gradient['template_data'] = compute_sobolev_gradient(
#             template_data.grad, self.smoothing_kernel_width, self.template).data.numpy()
#
# gradient_numpy = {}
# for (key, value) in gradient.items():
#     gradient_numpy[key] = value.data.numpy()
#     # print(key, value.data.numpy())
#
# # for i, target in enumerate(targets):
# #     self.diffeomorphism.set_initial_momenta(momenta[i])
# #     self.diffeomorphism.update()
# #     deformed_points = self.diffeomorphism.get_template_data()
# #     regularity -= self.diffeomorphism.get_norm_squared()
# #     attachment -= self.multi_object_attachment.compute_weighted_distance(
# #         deformed_points, self.template, target, self.objects_noise_variance)
# # #
# # return attachment, regularity
# return attachment.data.numpy()[0], regularity.data.numpy()[0], gradient_numpy