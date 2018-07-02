import logging

import torch
from torch.autograd import Variable

from core import default
from core.model_tools.deformations.geodesic import Geodesic
from core.models.abstract_statistical_model import AbstractStatisticalModel
from core.models.model_functions import create_regular_grid_of_points, compute_sobolev_gradient
from core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from in_out.array_readers_and_writers import *
from in_out.dataset_functions import create_template_metadata

logger = logging.getLogger(__name__)


class GeodesicRegression(AbstractStatisticalModel):
    """
    Geodesic regression object class.
    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, dataset, template_specifications, deformation_kernel,
                 concentration_of_time_points=default.concentration_of_time_points, t0=None,
                 number_of_time_points=default.number_of_time_points,
                 use_rk2_for_shoot=default.use_rk2_for_shoot, use_rk2_for_flow=default.use_rk2_for_flow,
                 initial_cp_spacing=default.initial_cp_spacing,
                 freeze_template=default.freeze_template,
                 freeze_control_points=default.freeze_control_points,
                 use_sobolev_gradient=default.use_sobolev_gradient,
                 smoothing_kernel_width=default.smoothing_kernel_width,
                 dense_mode=default.dense_mode,
                 number_of_threads=default.number_of_threads):
        AbstractStatisticalModel.__init__(self, name='GeodesicRegression')

        self.dataset = dataset

        if t0 is None:
            t0 = self.get_mean_visit_age(self.dataset.times)

        object_list, self.objects_name, self.objects_name_extension, self.objects_noise_variance, \
            self.multi_object_attachment = create_template_metadata(template_specifications, self.dataset.dimension, self.dataset.tensor_scalar_type)

        self.template = DeformableMultiObject(object_list, self.dataset.dimension)

        # self.multi_object_attachment = MultiObjectAttachment()
        self.geodesic = Geodesic(dimension=self.dataset.dimension, dense_mode=dense_mode, tensor_scalar_type=self.dataset.tensor_scalar_type,
                                 concentration_of_time_points=concentration_of_time_points, t0=t0,
                                 deformation_kernel=deformation_kernel, number_of_time_points=number_of_time_points,
                                 use_rk2_for_shoot=use_rk2_for_shoot, use_rk2_for_flow=use_rk2_for_flow)

        self.use_sobolev_gradient = use_sobolev_gradient
        self.smoothing_kernel_width = smoothing_kernel_width

        self.initial_cp_spacing = initial_cp_spacing
        self.number_of_objects = None
        self.number_of_control_points = None
        self.bounding_box = None

        # Dictionary of numpy arrays.
        self.fixed_effects['template_data'] = None
        self.fixed_effects['control_points'] = None
        self.fixed_effects['momenta'] = None

        self.freeze_template = freeze_template
        self.freeze_control_points = freeze_control_points

        self.dense_mode = dense_mode

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
        out['momenta'] = self.fixed_effects['momenta']
        return out

    def set_fixed_effects(self, fixed_effects):
        if not self.freeze_template:
            template_data = {key: fixed_effects[key] for key in self.fixed_effects['template_data'].keys()}
            self.set_template_data(template_data)
        if not self.freeze_control_points:
            self.set_control_points(fixed_effects['control_points'])
        self.set_momenta(fixed_effects['momenta'])

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def update(self):
        """
        Final initialization steps.
        """

        self.template.update(self.dataset.dimension)
        self.number_of_objects = len(self.template.object_list)
        self.bounding_box = self.template.bounding_box

        self.set_template_data(self.template.get_data())

        if self.fixed_effects['control_points'] is None:
            self._initialize_control_points()
        else:
            self._initialize_bounding_box()

        if self.fixed_effects['momenta'] is None:
            self._initialize_momenta()

    # Compute the functional. Numpy input/outputs.
    def compute_log_likelihood(self, population_RER, individual_RER, mode='complete', with_grad=False):
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
        template_data, template_points, control_points, momenta = self._fixed_effects_to_torch_tensors(with_grad)

        # Deform -------------------------------------------------------------------------------------------------------
        attachment, regularity = self._compute_attachment_and_regularity(template_data, template_points, control_points, momenta)

        # Compute gradient if needed -----------------------------------------------------------------------------------
        if with_grad:
            total = regularity + attachment
            total.backward()

            gradient = {}
            # Template data.
            if not self.freeze_template:
                if 'landmark_points' in template_data.keys():
                    gradient['landmark_points'] = template_points['landmark_points'].grad
                if 'image_intensities' in template_data.keys():
                    gradient['image_intensities'] = template_data['image_intensities'].grad
                # for key, value in template_data.items():
                #     gradient[key] = value.grad

                if self.use_sobolev_gradient and 'landmark_points' in gradient.keys():
                    gradient['landmark_points'] = compute_sobolev_gradient(
                        gradient['landmark_points'], self.smoothing_kernel_width, self.template, self.dataset.tensor_scalar_type)

            # Control points and momenta.
            if not self.freeze_control_points: gradient['control_points'] = control_points.grad
            gradient['momenta'] = momenta.grad

            # Convert the gradient back to numpy.
            gradient = {key: value.data.cpu().numpy() for key, value in gradient.items()}

            return attachment.detach().cpu().numpy(), regularity.detach().cpu().numpy(), gradient

        else:
            return attachment.detach().cpu().numpy(), regularity.detach().cpu().numpy()

    # def initialize_template_attributes(self, template_specifications):
    #     """
    #     Sets the Template, TemplateObjectsName, TemplateObjectsNameExtension, TemplateObjectsNorm,
    #     TemplateObjectsNormKernelType and TemplateObjectsNormKernelWidth attributes.
    #     """
    #
    #     t_list, t_name, t_name_extension, t_noise_variance, t_multi_object_attachment = \
    #         create_template_metadata(template_specifications)
    #
    #     self.template.object_list = t_list
    #     self.objects_name = t_name
    #     self.objects_name_extension = t_name_extension
    #     self.objects_noise_variance = t_noise_variance
    #     self.multi_object_attachment = t_multi_object_attachment

    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    def _compute_attachment_and_regularity(self, template_data, template_points, control_points, momenta):
        """
        Core part of the ComputeLogLikelihood methods. Fully torch.
        """

        # Initialize: cross-sectional dataset --------------------------------------------------------------------------
        target_times = self.dataset.times[0]
        target_objects = self.dataset.deformable_objects[0]

        # Deform -------------------------------------------------------------------------------------------------------
        self.geodesic.set_tmin(min(target_times))
        self.geodesic.set_tmax(max(target_times))
        self.geodesic.set_template_points_t0(template_points)
        self.geodesic.set_control_points_t0(control_points)
        self.geodesic.set_momenta_t0(momenta)
        self.geodesic.update()

        attachment = 0.
        for j, (time, obj) in enumerate(zip(target_times, target_objects)):
            deformed_points = self.geodesic.get_template_points(time)
            deformed_data = self.template.get_deformed_data(deformed_points, template_data)
            attachment -= self.multi_object_attachment.compute_weighted_distance(
                deformed_data, self.template, obj, self.objects_noise_variance)
        regularity = - self.geodesic.get_norm_squared()

        return attachment, regularity

    def _initialize_control_points(self):
        """
        Initialize the control points fixed effect.
        """
        if not self.dense_mode:
            control_points = create_regular_grid_of_points(self.bounding_box, self.initial_cp_spacing, self.dataset.dimension)
        else:
            assert (('landmark_points' in self.template.get_points().keys()) and
                    ('image_points' not in self.template.get_points().keys())), \
                'In dense mode, only landmark objects are allowed. One at least is needed.'
            control_points = self.template.get_points()['landmark_points']

        self.set_control_points(control_points)
        self.number_of_control_points = control_points.shape[0]
        logger.info('Set of ' + str(self.number_of_control_points) + ' control points defined.')

    def _initialize_momenta(self):
        """
        Initialize the momenta fixed effect.
        """
        momenta = np.zeros((self.number_of_control_points, self.dataset.dimension))
        self.set_momenta(momenta)

    def _initialize_bounding_box(self):
        """
        Initialize the bounding box. which tightly encloses all template objects and the atlas control points.
        Relevant when the control points are given by the user.
        """
        assert (self.number_of_control_points > 0)

        control_points = self.get_control_points()

        for k in range(self.number_of_control_points):
            for d in range(self.dataset.dimension):
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
        template_data = {key: Variable(torch.from_numpy(value).type(self.dataset.tensor_scalar_type),
                                       requires_grad=(not self.freeze_template and with_grad))
                         for key, value in template_data.items()}

        # Template points.
        template_points = self.template.get_points()
        template_points = {key: Variable(torch.from_numpy(value).type(self.dataset.tensor_scalar_type),
                                         requires_grad=(not self.freeze_template and with_grad))
                           for key, value in template_points.items()}

        # Control points.
        if self.dense_mode:
            assert (('landmark_points' in self.template.get_points().keys()) and
                    ('image_points' not in self.template.get_points().keys())), \
                'In dense mode, only landmark objects are allowed. One at least is needed.'
            control_points = template_points['landmark_points']
        else:
            control_points = self.fixed_effects['control_points']
            control_points = Variable(torch.from_numpy(control_points).type(self.dataset.tensor_scalar_type),
                                      requires_grad=((not self.freeze_control_points and with_grad)
                                                     or self.geodesic.get_kernel_type() == 'keops'))

        # Momenta.
        momenta = self.fixed_effects['momenta']
        momenta = Variable(torch.from_numpy(momenta).type(self.dataset.tensor_scalar_type), requires_grad=with_grad)

        return template_data, template_points, control_points, momenta

    def get_mean_visit_age(self, visit_ages):
        total_number_of_visits = 0
        mean_visit_age = 0.0
        for i in range(len(visit_ages)):
            for j in range(len(visit_ages[i])):
                total_number_of_visits += 1
                mean_visit_age += visit_ages[i][j]

        if total_number_of_visits > 0:
            mean_visit_age /= float(total_number_of_visits)

        return mean_visit_age

    ####################################################################################################################
    ### Writing methods:
    ####################################################################################################################

    def write(self, population_RER, individual_RER, output_dir, write_adjoint_parameters=False):
        self._write_model_predictions(output_dir, self.dataset, write_adjoint_parameters)
        self._write_model_parameters(output_dir)

    def _write_model_predictions(self, output_dir, dataset=None, write_adjoint_parameters=False):

        # Initialize ---------------------------------------------------------------------------------------------------
        template_data, template_points, control_points, momenta = self._fixed_effects_to_torch_tensors(False)
        target_times = dataset.times[0]

        # Deform -------------------------------------------------------------------------------------------------------
        self.geodesic.tmin = min(target_times)
        self.geodesic.tmax = max(target_times)
        self.geodesic.set_template_points_t0(template_points)
        self.geodesic.set_control_points_t0(control_points)
        self.geodesic.set_momenta_t0(momenta)
        self.geodesic.update()

        # Write --------------------------------------------------------------------------------------------------------
        # Geodesic flow.
        self.geodesic.write(self.name, self.objects_name, self.objects_name_extension, self.template, template_data,
                            output_dir, write_adjoint_parameters)

        # Model predictions.
        if dataset is not None:
            for j, time in enumerate(target_times):
                names = []
                for k, (object_name, object_extension) in enumerate(
                        zip(self.objects_name, self.objects_name_extension)):
                    name = self.name + '__Reconstruction__' + object_name + '__tp_' + str(j) + ('__age_%.2f' % time) \
                           + object_extension
                    names.append(name)
                deformed_points = self.geodesic.get_template_points(time)
                deformed_data = self.template.get_deformed_data(deformed_points, template_data)
                self.template.write(output_dir, names, {key: value.data.cpu().numpy() for key, value in deformed_data.items()})

    def _write_model_parameters(self, output_dir):
        # Template.
        template_names = []
        for k in range(len(self.objects_name)):
            aux = self.name + '__EstimatedParameters__Template_' + self.objects_name[k] + '__tp_' \
                  + str(self.geodesic.backward_exponential.number_of_time_points - 1) \
                  + ('__age_%.2f' % self.geodesic.t0) + self.objects_name_extension[k]
            template_names.append(aux)
        self.template.write(output_dir, template_names)

        # Control points.
        write_2D_array(self.get_control_points(), output_dir, self.name + "__EstimatedParameters__ControlPoints.txt")

        # Momenta.
        write_3D_array(self.get_momenta(), output_dir, self.name + "__EstimatedParameters__Momenta.txt")
