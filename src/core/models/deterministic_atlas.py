import torch
from torch.autograd import Variable
from concurrent.futures import ThreadPoolExecutor

from core.model_tools.attachments.multi_object_attachment import MultiObjectAttachment
from core.model_tools.deformations.exponential import Exponential
from core.models.abstract_statistical_model import AbstractStatisticalModel
from core.models.model_functions import create_regular_grid_of_points, compute_sobolev_gradient
from core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from in_out.array_readers_and_writers import *
from in_out.dataset_functions import create_template_metadata


def _subject_attachment_and_regularity(arg):
    """
    Auxiliary function for multithreading (cannot be a class method).
    """

    # Read arguments.
    (i, settings, template, template_data, control_points, momenta, freeze_template, freeze_control_points,
     freeze_momenta, target, multi_object_attachment, objects_noise_variance, exponential, with_grad,
     use_sobolev_gradient, smoothing_kernel_width) = arg
    Settings().initialize(settings)

    # Convert to torch tensors.
    template_data = {key: torch.from_numpy(value).requires_grad_(not freeze_template and with_grad).type(
        Settings().tensor_scalar_type) for key, value in template_data.items()}
    template_points = {key: torch.from_numpy(value).requires_grad_(not freeze_template and with_grad).type(
        Settings().tensor_scalar_type) for key, value in template.get_points().items()}
    control_points = torch.from_numpy(control_points).requires_grad_((
        not freeze_control_points and with_grad) or exponential.get_kernel_type() == 'keops').type(
        Settings().tensor_scalar_type)
    momenta = torch.from_numpy(momenta).requires_grad_(not freeze_momenta and with_grad).type(
        Settings().tensor_scalar_type)

    # Deform.
    exponential.set_initial_template_points(template_points)
    exponential.set_initial_control_points(control_points)
    exponential.set_initial_momenta(momenta)
    exponential.update()

    # Compute attachment and regularity.
    deformed_points = exponential.get_template_points()
    deformed_data = template.get_deformed_data(deformed_points, template_data)
    attachment = - multi_object_attachment.compute_weighted_distance(
        deformed_data, template, target, objects_noise_variance)
    regularity = - exponential.get_norm_squared()

    # Compute the gradient.
    if with_grad:
        total_for_subject = attachment + regularity
        total_for_subject.backward()

        gradient = {}
        if not freeze_template:
            if 'landmark_points' in template_data.keys():
                if use_sobolev_gradient:
                    gradient['landmark_points'] = compute_sobolev_gradient(
                        template_points['landmark_points'].grad.detach(),
                        smoothing_kernel_width, template).cpu().numpy()
                else:
                    gradient['landmark_points'] = template_points['landmark_points'].grad.detach().cpu().numpy()
            if 'image_intensities' in template_data.keys():
                gradient['image_intensities'] = template_data['image_intensities'].grad.detach().cpu().numpy()
        if not freeze_control_points:
            gradient['control_points'] = control_points.grad.detach().cpu().numpy()
        if not freeze_momenta:
            gradient['momenta'] = momenta.grad.detach().cpu().numpy()

        return i, attachment.detach().cpu().numpy(), regularity.detach().cpu().numpy(), gradient

    else:
        return i, attachment.detach().cpu().numpy(), regularity.detach().cpu().numpy()


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

        if Settings().number_of_threads > 1:
            targets = [target[0] for target in dataset.deformable_objects]
            args = [(i, Settings().serialize(), self.template, self.fixed_effects['template_data'],
                     self.fixed_effects['control_points'], self.fixed_effects['momenta'][i], self.freeze_template,
                     self.freeze_control_points, self.freeze_momenta, targets[i], self.multi_object_attachment,
                     self.objects_noise_variance, self.exponential.light_copy(), with_grad, self.use_sobolev_gradient,
                     self.smoothing_kernel_width) for i in range(len(targets))]

            # Perform parallelized computations.
            with ThreadPoolExecutor(max_workers=Settings().number_of_threads) as pool:
                results = pool.map(_subject_attachment_and_regularity, args)

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
                    i, attachment_i, regularity_i, gradient_i = result
                    attachment += attachment_i
                    regularity += regularity_i
                    for key, value in gradient_i.items():
                        if key == 'momenta': gradient[key][i] = value
                        else: gradient[key] += value
                return attachment, regularity, gradient
            else:
                attachment = 0.0
                regularity = 0.0
                for result in results:
                    i, attachment_i, regularity_i = result
                    attachment += attachment_i
                    regularity += regularity_i
                    return attachment, regularity

        else:
            template_data, template_points, control_points, momenta = self._fixed_effects_to_torch_tensors(with_grad)
            return self._compute_attachment_and_regularity(
                dataset, template_data, template_points, control_points, momenta, with_grad)


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
        Single-thread version.
        """
        # Initialize.
        targets = [target[0] for target in dataset.deformable_objects]

        regularity = 0.
        attachment = 0.

        # Deform.
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

        # Compute gradient.
        if with_grad:
            total = attachment + regularity
            total.backward()

            gradient = {}
            if not self.freeze_template:
                if 'landmark_points' in template_data.keys():
                    if self.use_sobolev_gradient:
                        gradient['landmark_points'] = compute_sobolev_gradient(
                            template_points['landmark_points'].grad.detach(),
                            self.smoothing_kernel_width, self.template).cpu().numpy()
                    else:
                        gradient['landmark_points'] = template_points['landmark_points'].grad.detach().cpu().numpy()
                if 'image_intensities' in template_data.keys():
                    gradient['image_intensities'] = template_data['image_intensities'].grad.detach().cpu().numpy()
            if not self.freeze_control_points:
                gradient['control_points'] = control_points.grad.detach().cpu().numpy()
            if not self.freeze_momenta:
                gradient['momenta'] = momenta.grad.detach().cpu().numpy()

            return attachment.detach().cpu().numpy(), regularity.detach().cpu().numpy(), gradient

        else:
            return attachment.detach().cpu().numpy(), regularity.detach().cpu().numpy()

    def _initialize_control_points(self):
        """
        Initialize the control points fixed effect.
        """
        if not Settings().dense_mode:
            control_points = create_regular_grid_of_points(self.bounding_box, self.initial_cp_spacing)
            for elt in self.template.object_list:
                if elt.type.lower() == 'image':
                    control_points = remove_useless_control_points(control_points, elt,
                                                                   self.exponential.get_kernel_width())
                    break
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
                                      requires_grad=((not self.freeze_control_points and with_grad)
                                                     or self.exponential.get_kernel_type() == 'keops'))
        # Momenta.
        momenta = self.fixed_effects['momenta']
        momenta = Variable(torch.from_numpy(momenta).type(Settings().tensor_scalar_type),
                           requires_grad=(not self.freeze_momenta and with_grad))

        return template_data, template_points, control_points, momenta

    ####################################################################################################################
    ### Writing methods:
    ####################################################################################################################

    def write(self, dataset, population_RER, individual_RER, write_residuals=True):

        # Write the model predictions, and compute the residuals at the same time.
        residuals = self._write_model_predictions(dataset, individual_RER, compute_residuals=write_residuals)

        # Write residuals.
        if write_residuals:
            residuals_list = [[residuals_i_k.data.cpu().numpy() for residuals_i_k in residuals_i]
                              for residuals_i in residuals]
            write_2D_list(residuals_list, self.name + "__EstimatedParameters__Residuals.txt")

        # Write the model parameters.
        self._write_model_parameters()

    def _write_model_predictions(self, dataset, individual_RER, compute_residuals=True):

        # Initialize.
        template_data, template_points, control_points, momenta = self._fixed_effects_to_torch_tensors(False)

        # Deform, write reconstructions and compute residuals.
        self.exponential.set_initial_template_points(template_points)
        self.exponential.set_initial_control_points(control_points)

        residuals = []  # List of torch 1D tensors. Individuals, objects.
        for i, subject_id in enumerate(dataset.subject_ids):
            self.exponential.set_initial_momenta(momenta[i])
            self.exponential.update()

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
            self.template.write(names, {key: value.data.cpu().numpy() for key, value in deformed_data.items()})

        return residuals

    def _write_model_parameters(self):

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

