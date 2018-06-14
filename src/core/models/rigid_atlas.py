import torch

from core.model_tools.attachments.multi_object_attachment import MultiObjectAttachment
from core.models.abstract_statistical_model import AbstractStatisticalModel
from core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from in_out.array_readers_and_writers import *
from in_out.dataset_functions import create_template_metadata

import logging

logger = logging.getLogger(__name__)


class RigidAtlas(AbstractStatisticalModel):
    """
    Rigid atlas object class.

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

        self.number_of_subjects = None
        self.number_of_objects = None

        # Dictionary of numpy arrays.
        self.fixed_effects['translations'] = None
        self.fixed_effects['rotations'] = None

    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################

    # Translations
    def get_translations(self):
        return self.fixed_effects['translations']

    def set_translations(self, t):
        self.fixed_effects['translations'] = t

    # Rotations
    def get_rotations(self):
        return self.fixed_effects['rotations']

    def set_rotations(self, r):
        self.fixed_effects['rotations'] = r

    # Full fixed effects
    def get_fixed_effects(self):
        out = {}
        out['translations'] = self.fixed_effects['translations']
        out['rotations'] = self.fixed_effects['rotations']
        return out

    def set_fixed_effects(self, fixed_effects):
        self.set_translations(fixed_effects['translations'])
        self.set_rotations(fixed_effects['rotations'])

    def get_fixed_effects_variability(self):
        out = {}
        out['translations'] = np.zeros(self.fixed_effects['translations'].shape) + 5.
        out['rotations'] = np.zeros(self.fixed_effects['rotations'].shape) + 0.5
        return out

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def update(self):
        """
        Final initialization steps.
        """

        self.template.update()
        self.number_of_objects = len(self.template.object_list)

        self._initialize_translations()
        self._initialize_rotations()

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

        # self.write(dataset, population_RER, individual_RER)

        translations = self.get_translations()
        rotations = self.get_rotations()

        template_points = {key: Settings().tensor_scalar_type(value)
                           for key, value in self.template.get_points().items()}
        template_data = {key: Settings().tensor_scalar_type(value)
                         for key, value in self.template.get_data().items()}

        attachment = 0.0
        targets = [target[0] for target in dataset.deformable_objects]

        if with_grad:
            gradient = {}
            gradient['translations'] = np.zeros(self.get_translations().shape)
            gradient['rotations'] = np.zeros(self.get_rotations().shape)

            for i, target in enumerate(targets):
                translation = Settings().tensor_scalar_type(translations[i]).requires_grad_(with_grad)
                rotation = Settings().tensor_scalar_type(rotations[i]).requires_grad_(with_grad)
                attachment_i = self._compute_subject_attachment(
                    template_points, template_data, translation, rotation, target)
                attachment -= attachment_i.detach().cpu().numpy()

                attachment_i.backward()
                gradient['translations'][i] = - translation.grad.detach().cpu().numpy()
                gradient['rotations'][i] = - rotation.grad.detach().cpu().numpy()

            return attachment, 0.0, gradient

        else:
            for i, target in enumerate(targets):
                translation = Settings().tensor_scalar_type(translations[i])
                rotation = Settings().tensor_scalar_type(rotations[i])
                attachment_i = self._compute_subject_attachment(
                    template_points, template_data, translation, rotation, target)
                attachment -= attachment_i.detach().cpu().numpy()
            return attachment, 0.0

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

    def _compute_subject_attachment(self, template_points, template_data, translation, rotation, target):
        rotation_matrix = self._compute_rotation_matrix(rotation)
        deformed_points = {key: torch.matmul(rotation_matrix.unsqueeze(0),
                                             value.unsqueeze(2)).view(-1, Settings().dimension)
                                + translation for key, value in template_points.items()}
        deformed_data = self.template.get_deformed_data(deformed_points, template_data)
        attachment = self.multi_object_attachment.compute_weighted_distance(
            deformed_data, self.template, target, self.objects_noise_variance)
        return attachment

    @staticmethod
    def _compute_rotation_matrix(rotation):

        if Settings().dimension == 2:
            raise RuntimeError('Not implemented yet.')

        elif Settings().dimension == 3:  # Using Euler angles: https://fr.wikipedia.org/wiki/Matrice_de_rotation
            psi = rotation[0]
            theta = rotation[0]
            phi = rotation[0]

            rot_x = torch.zeros(3, 3).type(Settings().tensor_scalar_type)
            rot_x[0, 0] = phi.cos()
            rot_x[0, 1] = - phi.sin()
            rot_x[1, 0] = phi.sin()
            rot_x[1, 1] = phi.cos()
            rot_x[2, 2] = 1.

            rot_y = torch.zeros(3, 3).type(Settings().tensor_scalar_type)
            rot_y[0, 0] = theta.cos()
            rot_y[0, 2] = theta.sin()
            rot_y[2, 0] = - theta.sin()
            rot_y[2, 2] = theta.cos()
            rot_y[1, 1] = 1.

            rot_z = torch.zeros(3, 3).type(Settings().tensor_scalar_type)
            rot_z[1, 1] = psi.cos()
            rot_z[1, 2] = - psi.sin()
            rot_z[2, 1] = psi.sin()
            rot_z[2, 2] = psi.cos()
            rot_z[0, 0] = 1.

            rotation_matrix = torch.mm(rot_z, torch.mm(rot_y, rot_x))
            return rotation_matrix

    def _initialize_translations(self):
        assert (self.number_of_subjects > 0)
        if self.get_translations() is None:
            translations = np.zeros((self.number_of_subjects, Settings().dimension))
            self.set_translations(translations)
            logger.info('Translations initialized to zero, for ' + str(self.number_of_subjects) + ' subjects.')

    def _initialize_rotations(self):
        assert (self.number_of_subjects > 0)
        if self.get_rotations() is None:
            rotations = np.zeros((self.number_of_subjects, Settings().dimension))
            self.set_rotations(rotations)
            logger.info('Rotation Euler angles initialized to zero, for ' + str(self.number_of_subjects) + ' subjects.')

    ####################################################################################################################
    ### Private utility methods:
    ####################################################################################################################

    def _fixed_effects_to_torch_tensors(self, with_grad):
        """
        Convert the fixed_effects into torch tensors.
        """
        translations = Settings().tensor_scalar_type(self.get_translations()).requires_grad_(with_grad)
        rotations = Settings().tensor_scalar_type(self.get_rotations()).requires_grad_(with_grad)

        return translations, rotations

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
        translations = self.get_translations()
        rotations = self.get_rotations()

        template_points = {key: Settings().tensor_scalar_type(value)
                           for key, value in self.template.get_points().items()}
        template_data = {key: Settings().tensor_scalar_type(value)
                         for key, value in self.template.get_data().items()}

        targets = [target[0] for target in dataset.deformable_objects]
        residuals = []

        for i, (subject_id, target) in enumerate(zip(dataset.subject_ids, targets)):
            translation = Settings().tensor_scalar_type(translations[i])
            rotation = Settings().tensor_scalar_type(rotations[i])
            rotation_matrix = self._compute_rotation_matrix(rotation)

            # Direct deformation ---------------------------------------------------------------------------------------
            deformed_points = {key: torch.matmul(rotation_matrix.unsqueeze(0),
                                                 value.unsqueeze(2)).view(-1, Settings().dimension) + translation
                               for key, value in template_points.items()}
            deformed_data = self.template.get_deformed_data(deformed_points, template_data)

            if compute_residuals:
                residuals.append(self.multi_object_attachment.compute_distances(deformed_data, self.template, target))

            names = []
            for k, (object_name, object_extension) \
                    in enumerate(zip(self.objects_name, self.objects_name_extension)):
                name = self.name + '__Reconstruction__' + object_name + '__subject_' + subject_id + object_extension
                names.append(name)
            self.template.write(names, {key: value.data.cpu().numpy() for key, value in deformed_data.items()})

            # Inverse deformation --------------------------------------------------------------------------------------
            target_points = {key: Settings().tensor_scalar_type(value) for key, value in target.get_points().items()}
            target_data = {key: Settings().tensor_scalar_type(value) for key, value in target.get_data().items()}

            deformed_points = {key: torch.matmul(rotation_matrix.t().unsqueeze(0),
                                                 (value - translation).unsqueeze(2)).view(-1, Settings().dimension)
                               for key, value in target_points.items()}
            deformed_data = target.get_deformed_data(deformed_points, target_data)

            names = []
            for k, (object_name, object_extension) \
                    in enumerate(zip(self.objects_name, self.objects_name_extension)):
                name = self.name + '__Registration__' + object_name + '__subject_' + subject_id + object_extension
                names.append(name)
            target.write(names, {key: value.data.cpu().numpy() for key, value in deformed_data.items()})

        return residuals

    def _write_model_parameters(self):
        write_2D_array(self.get_translations(), self.name + "__EstimatedParameters__Translations.txt")
        write_3D_array(self.get_rotations(), self.name + "__EstimatedParameters__Rotations.txt")
