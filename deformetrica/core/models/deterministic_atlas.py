import math
import time

import torch

from ...support import kernels as kernel_factory
from ...core import default
from ...core.model_tools.deformations.exponential import Exponential
from ...core.models.abstract_statistical_model import AbstractStatisticalModel
from ...core.models.model_functions import initialize_control_points, initialize_momenta
from ...core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from ...in_out.array_readers_and_writers import *
from ...in_out.dataset_functions import create_template_metadata
from ...support import utilities

logger = logging.getLogger(__name__)


def _subject_attachment_and_regularity(arg):
    """
    Auxiliary function for multithreading (cannot be a class method).
    """
    from .abstract_statistical_model import process_initial_data
    if process_initial_data is None:
        raise RuntimeError('process_initial_data is not set !')

    # Read arguments.
    (deformable_objects, multi_object_attachment, objects_noise_variance,
     freeze_template, freeze_control_points, freeze_momenta,
     exponential, sobolev_kernel, use_sobolev_gradient, tensor_scalar_type, gpu_mode) = process_initial_data
    (i, template, template_data, control_points, momenta, with_grad) = arg

    # start = time.perf_counter()
    device, device_id = utilities.get_best_device(gpu_mode=gpu_mode)
    # device, device_id = ('cpu', -1)
    if device_id >= 0:
        torch.cuda.set_device(device_id)

    # convert np.ndarrays to torch tensors. This is faster than transferring torch tensors to process.
    template_data = {key: utilities.move_data(value, device=device, dtype=tensor_scalar_type,
                                              requires_grad=with_grad and not freeze_template)
                     for key, value in template_data.items()}
    template_points = {key: utilities.move_data(value, device=device, dtype=tensor_scalar_type,
                                                requires_grad=with_grad and not freeze_template)
                       for key, value in template.get_points().items()}
    control_points = utilities.move_data(control_points, device=device, dtype=tensor_scalar_type,
                                         requires_grad=with_grad and not freeze_control_points)
    momenta = utilities.move_data(momenta, device=device, dtype=tensor_scalar_type,
                                  requires_grad=with_grad and not freeze_momenta)

    assert torch.device(
        device) == control_points.device == momenta.device, 'control_points and momenta tensors must be on the same device. ' \
                                                            'device=' + device + \
                                                            ', control_points.device=' + str(control_points.device) + \
                                                            ', momenta.device=' + str(momenta.device)

    attachment, regularity = DeterministicAtlas._deform_and_compute_attachment_and_regularity(
        exponential, template_points, control_points, momenta,
        template, template_data, multi_object_attachment,
        deformable_objects[i], objects_noise_variance,
        device)

    res = DeterministicAtlas._compute_gradients(
        attachment, regularity, template_data,
        freeze_template, template_points,
        freeze_control_points, control_points,
        freeze_momenta, momenta,
        use_sobolev_gradient, sobolev_kernel,
        with_grad)
    # elapsed = time.perf_counter() - start
    # logger.info('pid=' + str(os.getpid()) + ', ' + torch.multiprocessing.current_process().name +
    #       ', device=' + device + ', elapsed=' + str(elapsed))
    return i, res


class DeterministicAtlas(AbstractStatisticalModel):
    """
    Deterministic atlas object class.

    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, template_specifications, number_of_subjects,

                 dimension=default.dimension,
                 tensor_scalar_type=default.tensor_scalar_type,
                 tensor_integer_type=default.tensor_integer_type,
                 dense_mode=default.dense_mode,
                 number_of_processes=default.number_of_processes,

                 deformation_kernel_type=default.deformation_kernel_type,
                 deformation_kernel_width=default.deformation_kernel_width,
                 deformation_kernel_device=default.deformation_kernel_device,

                 shoot_kernel_type=default.shoot_kernel_type,
                 number_of_time_points=default.number_of_time_points,
                 use_rk2_for_shoot=default.use_rk2_for_shoot, use_rk2_for_flow=default.use_rk2_for_flow,

                 freeze_template=default.freeze_template,
                 use_sobolev_gradient=default.use_sobolev_gradient,
                 smoothing_kernel_width=default.smoothing_kernel_width,

                 initial_control_points=default.initial_control_points,
                 freeze_control_points=default.freeze_control_points,
                 initial_cp_spacing=default.initial_cp_spacing,

                 initial_momenta=default.initial_momenta,
                 freeze_momenta=default.freeze_momenta,

                 gpu_mode=default.gpu_mode,
                 process_per_gpu=default.process_per_gpu,

                 **kwargs):

        AbstractStatisticalModel.__init__(self, name='DeterministicAtlas', number_of_processes=number_of_processes,
                                          gpu_mode=gpu_mode)

        # Global-like attributes.
        self.dimension = dimension
        self.tensor_scalar_type = tensor_scalar_type
        self.tensor_integer_type = tensor_integer_type
        self.dense_mode = dense_mode

        # Declare model structure.
        self.fixed_effects['template_data'] = None
        self.fixed_effects['control_points'] = None
        self.fixed_effects['momenta'] = None

        self.freeze_template = freeze_template
        self.freeze_control_points = freeze_control_points
        self.freeze_momenta = freeze_momenta

        # Deformation.
        self.exponential = Exponential(
            dense_mode=dense_mode,
            kernel=kernel_factory.factory(deformation_kernel_type,
                                          gpu_mode=gpu_mode,
                                          kernel_width=deformation_kernel_width),
            shoot_kernel_type=shoot_kernel_type,
            number_of_time_points=number_of_time_points,
            use_rk2_for_shoot=use_rk2_for_shoot, use_rk2_for_flow=use_rk2_for_flow)

        # Template.
        (object_list, self.objects_name, self.objects_name_extension,
         self.objects_noise_variance, self.multi_object_attachment) = create_template_metadata(template_specifications,
                                                                                               self.dimension)

        self.template = DeformableMultiObject(object_list)
        # self.template.update()

        self.number_of_objects = len(self.template.object_list)

        self.use_sobolev_gradient = use_sobolev_gradient
        self.smoothing_kernel_width = smoothing_kernel_width
        if self.use_sobolev_gradient:
            self.sobolev_kernel = kernel_factory.factory(deformation_kernel_type,
                                                         gpu_mode=gpu_mode,
                                                         kernel_width=smoothing_kernel_width)

        # Template data.
        self.fixed_effects['template_data'] = self.template.get_data()

        # Control points.
        self.fixed_effects['control_points'] = initialize_control_points(
            initial_control_points, self.template, initial_cp_spacing, deformation_kernel_width,
            self.dimension, self.dense_mode)
        self.number_of_control_points = len(self.fixed_effects['control_points'])

        # Momenta.
        self.fixed_effects['momenta'] = initialize_momenta(
            initial_momenta, self.number_of_control_points, self.dimension, number_of_subjects)
        self.number_of_subjects = number_of_subjects

        self.process_per_gpu = process_per_gpu

    def initialize_noise_variance(self, dataset, device='cpu'):
        if np.min(self.objects_noise_variance) < 0:
            template_data, template_points, control_points, momenta = self._fixed_effects_to_torch_tensors(False,
                                                                                                           device=device)
            targets = dataset.deformable_objects
            targets = [target[0] for target in targets]

            residuals_torch = []
            self.exponential.set_initial_template_points(template_points)
            self.exponential.set_initial_control_points(control_points)
            for i, target in enumerate(targets):
                self.exponential.set_initial_momenta(momenta[i])
                self.exponential.update()
                deformed_points = self.exponential.get_template_points()
                deformed_data = self.template.get_deformed_data(deformed_points, template_data)
                residuals_torch.append(self.multi_object_attachment.compute_distances(
                    deformed_data, self.template, target))

            residuals = np.zeros((self.number_of_objects,))
            for i in range(len(residuals_torch)):
                residuals += residuals_torch[i].detach().cpu().numpy()

            # Initialize the noise variance hyper-parameter as a 1/100th of the initial residual.
            for k, obj in enumerate(self.objects_name):
                if self.objects_noise_variance[k] < 0:
                    nv = 0.01 * residuals[k] / float(self.number_of_subjects)
                    self.objects_noise_variance[k] = nv
                    logger.info('>> Automatically chosen noise std: %.4f [ %s ]' % (math.sqrt(nv), obj))

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
        # self.number_of_control_points = len(cp)

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

    def setup_multiprocess_pool(self, dataset):
        self._setup_multiprocess_pool(initargs=([target[0] for target in dataset.deformable_objects],
                                                self.multi_object_attachment,
                                                self.objects_noise_variance,
                                                self.freeze_template, self.freeze_control_points, self.freeze_momenta,
                                                self.exponential, self.sobolev_kernel, self.use_sobolev_gradient,
                                                self.tensor_scalar_type, self.gpu_mode))

    # Compute the functional. Numpy input/outputs.
    def compute_log_likelihood(self, dataset, population_RER, individual_RER, mode='complete', with_grad=False):
        """
        Compute the log-likelihood of the dataset, given parameters fixed_effects and random effects realizations
        population_RER and indRER.

        :param fixed_effects: Dictionary of fixed effects.
        :param population_RER: Dictionary of population random effects realizations.
        :param individual_RER: Dictionary of individual random effects realizations.
        :param mode: Indicates which log_likelihood should be computed, between 'complete', 'model', and 'class2'.
        :param with_grad: Flag that indicates wether the gradient should be returned as well.
        :return:
        """

        if self.number_of_processes > 1:
            targets = [target[0] for target in dataset.deformable_objects]
            args = [(i, self.template,
                     self.fixed_effects['template_data'],
                     self.fixed_effects['control_points'],
                     self.fixed_effects['momenta'][i],
                     with_grad) for i in range(len(targets))]

            start = time.perf_counter()
            results = self.pool.map(_subject_attachment_and_regularity, args, chunksize=1)  # TODO: optimized chunk size
            # results = self.pool.imap_unordered(_subject_attachment_and_regularity, args, chunksize=1)
            # results = self.pool.imap(_subject_attachment_and_regularity, args, chunksize=int(len(args)/self.number_of_processes))
            logger.debug('time taken for deformations : ' + str(time.perf_counter() - start))

            # Sum and return.
            if with_grad:
                attachment = 0.0
                regularity = 0.0

                gradient = {}
                if not self.freeze_template:
                    for key, value in self.fixed_effects['template_data'].items():
                        gradient[key] = np.zeros(value.shape)
                if not self.freeze_control_points:
                    gradient['control_points'] = np.zeros(self.fixed_effects['control_points'].shape)
                if not self.freeze_momenta:
                    gradient['momenta'] = np.zeros(self.fixed_effects['momenta'].shape)

                for result in results:
                    i, (attachment_i, regularity_i, gradient_i) = result
                    attachment += attachment_i
                    regularity += regularity_i
                    for key, value in gradient_i.items():
                        if key == 'momenta':
                            gradient[key][i] = value
                        else:
                            gradient[key] += value
                return attachment, regularity, gradient
            else:
                attachment = 0.0
                regularity = 0.0
                for result in results:
                    i, (attachment_i, regularity_i) = result
                    attachment += attachment_i
                    regularity += regularity_i
                return attachment, regularity

        else:
            device, device_id = utilities.get_best_device(gpu_mode=self.gpu_mode)
            template_data, template_points, control_points, momenta = self._fixed_effects_to_torch_tensors(with_grad,
                                                                                                           device=device)
            return self._compute_attachment_and_regularity(dataset, template_data, template_points, control_points,
                                                           momenta, with_grad, device=device)

    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    @staticmethod
    def _deform_and_compute_attachment_and_regularity(exponential, template_points, control_points, momenta,
                                                      template, template_data,
                                                      multi_object_attachment, deformable_objects,
                                                      objects_noise_variance,
                                                      device='cpu'):
        # Deform.
        exponential.set_initial_template_points(template_points)
        exponential.set_initial_control_points(control_points)
        exponential.set_initial_momenta(momenta)
        exponential.move_data_to_(device=device)
        exponential.update()

        # Compute attachment and regularity.
        deformed_points = exponential.get_template_points()
        deformed_data = template.get_deformed_data(deformed_points, template_data)
        attachment = -multi_object_attachment.compute_weighted_distance(deformed_data, template, deformable_objects,
                                                                        objects_noise_variance)
        regularity = -exponential.get_norm_squared()

        assert torch.device(
            device) == attachment.device == regularity.device, 'attachment and regularity tensors must be on the same device. ' \
                                                               'device=' + device + \
                                                               ', attachment.device=' + str(attachment.device) + \
                                                               ', regularity.device=' + str(regularity.device)

        return attachment, regularity

    @staticmethod
    def _compute_gradients(attachment, regularity, template_data,
                           freeze_template, template_points,
                           freeze_control_points, control_points,
                           freeze_momenta, momenta,
                           use_sobolev_gradient, sobolev_kernel,
                           with_grad=False):
        if with_grad:
            total_for_subject = attachment + regularity
            total_for_subject.backward()

            gradient = {}
            if not freeze_template:
                if 'landmark_points' in template_data.keys():
                    assert template_points['landmark_points'].grad is not None, 'Gradients have not been computed'
                    if use_sobolev_gradient:
                        gradient['landmark_points'] = sobolev_kernel.convolve(
                            template_data['landmark_points'].detach(), template_data['landmark_points'].detach(),
                            template_points['landmark_points'].grad.detach()).cpu().numpy()
                    else:
                        gradient['landmark_points'] = template_points['landmark_points'].grad.detach().cpu().numpy()
                if 'image_intensities' in template_data.keys():
                    assert template_data['image_intensities'].grad is not None, 'Gradients have not been computed'
                    gradient['image_intensities'] = template_data['image_intensities'].grad.detach().cpu().numpy()
            if not freeze_control_points:
                assert control_points.grad is not None, 'Gradients have not been computed'
                gradient['control_points'] = control_points.grad.detach().cpu().numpy()
            if not freeze_momenta:
                assert momenta.grad is not None, 'Gradients have not been computed'
                gradient['momenta'] = momenta.grad.detach().cpu().numpy()

            res = attachment.detach().cpu().numpy(), regularity.detach().cpu().numpy(), gradient

        else:
            res = attachment.detach().cpu().numpy(), regularity.detach().cpu().numpy()

        return res

    def _compute_attachment_and_regularity(self, dataset, template_data, template_points, control_points, momenta,
                                           with_grad=False, device='cpu'):
        """
        Core part of the ComputeLogLikelihood methods. Torch input, numpy output_4_low_level.
        Single-thread version.
        """

        # Initialize.
        targets = [target[0] for target in dataset.deformable_objects]
        attachment = 0.
        regularity = 0.

        # loop for every deformable object
        # deform and update attachment and regularity
        for i, target in enumerate(targets):
            new_attachment, new_regularity = DeterministicAtlas._deform_and_compute_attachment_and_regularity(
                self.exponential, template_points, control_points, momenta[i],
                self.template, template_data, self.multi_object_attachment,
                target, self.objects_noise_variance,
                device=device)

            attachment += new_attachment
            regularity += new_regularity

        # Compute gradient.
        return self._compute_gradients(attachment, regularity, template_data,
                                       self.freeze_template, template_points,
                                       self.freeze_control_points, control_points,
                                       self.freeze_momenta, momenta,
                                       self.use_sobolev_gradient, self.sobolev_kernel,
                                       with_grad)

    ####################################################################################################################
    ### Private utility methods:
    ####################################################################################################################

    def _fixed_effects_to_torch_tensors(self, with_grad, device='cpu'):
        """
        Convert the fixed_effects into torch tensors.
        """
        # Template data.
        template_data = self.fixed_effects['template_data']
        template_data = {key: utilities.move_data(value, device=device, dtype=self.tensor_scalar_type,
                                                  requires_grad=(not self.freeze_template and with_grad))
                         for key, value in template_data.items()}
        # template_data = {key: Variable(torch.from_numpy(value).type(self.tensor_scalar_type),
        #                                requires_grad=(not self.freeze_template and with_grad))
        #                  for key, value in template_data.items()}

        # Template points.
        template_points = self.template.get_points()
        template_points = {key: utilities.move_data(value, device=device, dtype=self.tensor_scalar_type,
                                                    requires_grad=(not self.freeze_template and with_grad))
                           for key, value in template_points.items()}
        # template_points = {key: Variable(torch.from_numpy(value).type(self.tensor_scalar_type),
        #                                  requires_grad=(not self.freeze_template and with_grad))
        #                    for key, value in template_points.items()}

        # Control points.
        if self.dense_mode:
            assert (('landmark_points' in self.template.get_points().keys()) and
                    ('image_points' not in self.template.get_points().keys())), \
                'In dense mode, only landmark objects are allowed. One at least is needed.'
            control_points = template_points['landmark_points']
        else:
            control_points = self.fixed_effects['control_points']
            control_points = utilities.move_data(control_points, device=device, dtype=self.tensor_scalar_type,
                                                 requires_grad=(not self.freeze_control_points and with_grad))
            # control_points = Variable(torch.from_numpy(control_points).type(self.tensor_scalar_type),
            #                           requires_grad=(not self.freeze_control_points and with_grad))
        # Momenta.
        momenta = self.fixed_effects['momenta']
        momenta = utilities.move_data(momenta, device=device, dtype=self.tensor_scalar_type,
                                      requires_grad=(not self.freeze_momenta and with_grad))
        # momenta = Variable(torch.from_numpy(momenta).type(self.tensor_scalar_type),
        #                    requires_grad=(not self.freeze_momenta and with_grad))

        return template_data, template_points, control_points, momenta

    ####################################################################################################################
    ### Writing methods:
    ####################################################################################################################

    def write(self, dataset, population_RER, individual_RER, output_dir, write_residuals=True):

        # Write the model predictions, and compute the residuals at the same time.
        residuals = self._write_model_predictions(dataset, individual_RER, output_dir,
                                                  compute_residuals=write_residuals)

        # Write residuals.
        if write_residuals:
            residuals_list = [[residuals_i_k.data.cpu().numpy() for residuals_i_k in residuals_i]
                              for residuals_i in residuals]
            write_2D_list(residuals_list, output_dir, self.name + "__EstimatedParameters__Residuals.txt")

        # Write the model parameters.
        self._write_model_parameters(output_dir)

    def _write_model_predictions(self, dataset, individual_RER, output_dir, compute_residuals=True):
        device, _ = utilities.get_best_device(self.gpu_mode)

        # Initialize.
        template_data, template_points, control_points, momenta = self._fixed_effects_to_torch_tensors(False, device=device)

        # Deform, write reconstructions and compute residuals.
        self.exponential.set_initial_template_points(template_points)
        self.exponential.set_initial_control_points(control_points)

        residuals = []  # List of torch 1D tensors. Individuals, objects.
        for i, subject_id in enumerate(dataset.subject_ids):
            self.exponential.set_initial_momenta(momenta[i])
            self.exponential.update()

            # Writing the whole flow.
            names = []
            for k, object_name in enumerate(self.objects_name):
                name = self.name + '__flow__' + object_name + '__subject_' + subject_id
                names.append(name)
            self.exponential.write_flow(names, self.objects_name_extension, self.template, template_data, output_dir)

            deformed_points = self.exponential.get_template_points()
            deformed_data = self.template.get_deformed_data(deformed_points, template_data)

            if compute_residuals:
                residuals.append(self.multi_object_attachment.compute_distances(
                    deformed_data, self.template, dataset.deformable_objects[i][0]))

            names = []
            for k, (object_name, object_extension) \
                    in enumerate(zip(self.objects_name, self.objects_name_extension)):
                name = self.name + '__Reconstruction__' + object_name + '__subject_' + subject_id + object_extension
                names.append(name)
            self.template.write(output_dir, names,
                                {key: value.data.cpu().numpy() for key, value in deformed_data.items()})

        return residuals

    def _write_model_parameters(self, output_dir):

        # Template.
        template_names = []
        for i in range(len(self.objects_name)):
            aux = self.name + "__EstimatedParameters__Template_" + self.objects_name[i] + self.objects_name_extension[i]
            template_names.append(aux)
        self.template.write(output_dir, template_names)

        # Control points.
        write_2D_array(self.get_control_points(), output_dir, self.name + "__EstimatedParameters__ControlPoints.txt")

        # Momenta.
        write_3D_array(self.get_momenta(), output_dir, self.name + "__EstimatedParameters__Momenta.txt")
